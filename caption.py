import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import os
# from scipy.misc import imread, imresize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_image(image_path):

    # img = imread(image_path)
    img = plt.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    # img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize(size=(256, 256)))

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    return image


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    ######################################################################################
    # (1) and (2) - read and process an image
    ######################################################################################

    # (1) Read image and process
    image = _read_image(image_path)

    # (2a) Encode image
    # (1, 3, 256, 256)
    image = image.unsqueeze(0)  
    # (1, enc_image_size, enc_image_size, encoder_dim)
    # (1, 14, 14, 2048)
    encoder_out = encoder(image)  
    # 14
    enc_image_size = encoder_out.size(1)
    # 2048
    encoder_dim = encoder_out.size(3) 

    # (2b) Flatten encoding
    # (1, 196, 2048)
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    # 196
    num_pixels = encoder_out.size(1)

    ######################################################################################
    # (3) - prepare for BEAM search and the main loop
    ######################################################################################

    # We'll treat the problem as having a batch size of k
    # we just repeat encoder_out k times
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    # tensor([[9488],
    #         [9488],
    #         [9488]])
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    # we use step to limit the length of generated sequence
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    ######################################################################################
    # (4) main loop
    ######################################################################################

    rev_word_map = {v: k for k, v in word_map.items()}

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        # DEBUGGING
        # print(f'k_prev_words:{k_prev_words.shape} top_k_scores:{top_k_scores.shape} seqs:{seqs.shape}')
        # words = [rev_word_map[int(i)] for i in k_prev_words.reshape(-1)]
        # print(f'step:{step} k_prev_words:{k_prev_words.reshape(-1)} {words}')

        # print(f'step:{step}')
        # for i in range(k):
        #     print(' '.join([rev_word_map[int(j)] for j in seqs[i]]))
        # print()

        # print(f'top_k_scores:{top_k_scores.reshape(-1).detach().numpy()}')

        # DEBUGGING

        # (4-01) run LSTM cell as in forward()
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        # (4-02) compute scores (in forward we don't compute them)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # (4-03) Add top_k_scores
        # when step = 1 top_k_scores contains 0 and we don't change scores
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)


        # (4-04) Choose top_k_scores
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        # topk() Returns the k largest elements of the given input tensor along a given dimension.
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k=k, dim=0, largest=True, sorted=True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k=k, dim=0, largest=True, sorted=True)  # (s)

        # (4-05) Convert unrolled indices to actual indices of scores
        # prev_word_inds = top_k_words // vocab_size # (s)
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='trunc')
        # we choose top_k_words from range(3 * vocab_size) 
        # so we need to get a number  from range(vocab_size)
        next_word_inds = top_k_words % vocab_size  # (s)

        # if step == 3:
        #     print(f'step:{step}')
        #     print(f'k_prev_words: {caps_to_string(k_prev_words.reshape(-1), rev_word_map)}')
        #     print(f'next_word_inds: {caps_to_string(next_word_inds, rev_word_map)}')

        # (4-06) Add new words to sequences, alphas

        # if step == 3:
            # print(f'step:{step}')
            # print(f'prev_word_inds:{prev_word_inds.reshape(-1)}')
            # print(f'next_word_inds:{next_word_inds.reshape(-1)} {caps_to_string(next_word_inds.reshape(-1), rev_word_map)}')

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # (4-07) Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # (4-08) Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # (4-09) Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # (5) Choose final sequence
    i = complete_seqs_scores.index(max(complete_seqs_scores))

    # print(f'complete_seqs_scores:{[np.around(score.item(), 2) for score in complete_seqs_scores]} i:{i}')
    # print(f'complete_seqs:{[caps_to_string(s, rev_word_map) for s in complete_seqs]}')

    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def caps_to_string(s, rev_word_map):
    return ' '.join([rev_word_map[int(i)] for i in s])


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # print(f'words:{words}')

    _visualize(words, image, alphas)


def _visualize(words, image, alphas, smooth=True):

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        # show next word from words
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)

        # show image
        plt.imshow(image)

        #######################################################################################
        # The magic goes here
        #######################################################################################        
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        #######################################################################################

        plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
