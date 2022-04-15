import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            output_size=(encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # why do we need to permute this?
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, 
        # default value from train.py
        fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, 
        encoder_dim=2048, 
        decoder_dim=512, 
        attention_dim=512):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)  
        # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (batch_size, num_pixels)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  

        # (batch_size, num_pixels)
        alpha = self.softmax(att)  

        # (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, 

        # all sizes are from train.py file 

        attention_dim=512, 

        # size of embedding layer
        embed_dim=512, 

        # size of the hidden state ht
        decoder_dim=512, 

        # flickr8k size with min_freq = 5
        vocab_size=2633, 

        # not quite clear why do we use 2048? 
        # if we get (14, 14, 2048)? maybe we use projection?
        encoder_dim=2048, 

        dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # attention network
        self.attention = Attention(
            encoder_dim=encoder_dim, 
            decoder_dim=decoder_dim, 
            attention_dim=attention_dim)  

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)  # embedding layer

        # initial states
        # (BS, encoder_dim) -> (BS, decoder_dim)
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  

        self.dropout = nn.Dropout(p=self.dropout)

        # this is our LSTM; we use cell in a loop
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell



        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        # (batch_size, num_pixels, encoder_dim) -> (batch_size, encoder_dim)
        # encoder_dim = 2048
        mean_encoder_out = encoder_out.mean(dim=1)

        # (batch_size, encoder_dim) -> (batch_size, decoder_dim)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        ################################################################################################
        # Step (1): getting Encoder output; reshape and sort it;
        ################################################################################################

        batch_size = encoder_out.size(0)
        # encoder_out: (1, 14, 14, 2048) so encoder_dim = 2048
        encoder_dim = encoder_out.size(-1)
        # in our case 2633
        vocab_size = self.vocab_size

        # Flatten image: so we need this 196 pixels for some purpose
        # (batch_size, num_pixels, encoder_dim) 
        # in our case: (batch_size, 196, 2048) where 196 = 14 * 14
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        ################################################################################################
        # Step (2): Getting embeddings and initial LSTM state
        ################################################################################################

        # Embedding
        # (batch_size, max_caption_length) -> 
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        # encoder_out is AFTER modification: (batch_size, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        # caption_lengths - 1: broadcasting on a tensor
        decode_lengths = (caption_lengths - 1).tolist()

        ################################################################################################
        # Step (3): Main loop of LSTM Decoder
        ################################################################################################

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])

            # compute attention; they use a gate - see tutorial; we get back encoder_dim = 2048
            # (batch_size_t, num_pixels, encoder_dim), (batch_size_t, decoder_dim) -> (batch_size_t, encoder_dim)
            # (batch_size_t, 196, 2048), (batch_size_t, 512) -> (batch_size_t, 2048) 
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, 
            attention_weighted_encoding = gate * attention_weighted_encoding

            # decode_step - LSTM cell
            # (batch_size_t, embed_dim + encoder_dim) -> (batch_size_t, decoder_dim)
            # so we use embedded captions (only a single word at a step) and ourput of 
            # our encoder after attention and translate this with LSTM cell into hidden state
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )  

            # this projection from hidden space onto vocabulary space
            # (batch_size_t, decoder_dim) -> (batch_size_t, vocab_size)
            # we don't compute softmax here or choose max value
            preds = self.fc(self.dropout(h)) 
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # not quite clear why we need all these variables
        # encoded_captions - sorted captions (BS, max_caps_len)
        # sort_ind - index to sort images and captions (BS,)
        # decode_lengths - length of (BS,)

        # alphas - we use them in loss correction (BS, max_caps_len, num_pixels) 
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind








