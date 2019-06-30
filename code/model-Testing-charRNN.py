#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:10:48 2019

@author: macbook
"""


path = "../model/generative_charrnn_lr_"+str(0.0025)+"_hidden_layers"+str(128)+"_nlayers_"+str(1)
n_characters = len(string.printable)
model = RNN(n_characters, 128, n_characters)
model.load_state_dict(torch.load(path))
# model.eval()

def unicode_to_ascii(s):
    all_characters = string.printable
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_characters
    )
    
sim_txt = 'You shall use the site for lawful purposes only. You shall not post or transmit through the site any material which violates or infringes in any way upon the rights of others, which is unlawful, threatening, abusive, defamatory, invasive of privacy or publicity rights, vulgar, obscene, profane or otherwise objectionable, which encourages conduct that would constitute a criminal offense, gives rise to civil liability or otherwise violate any law.'
dissim_txt = 'Alice had no idea what to do, and in despair she put her hand into her pocket and pulled out a box of comfits (luckily the salt-water had not got into it) and handed them round as prizes. There was exactly one a-piece, all round. "Oh, Ive had such a curious dream!" said Alice. And she told hersister, as well as she could remember them, all these strange adventures of hers that you have just been reading about. Alice got up and ran off, thinking while she ran, as well she might, what a wonderful dream it had been.'
v_dissim_txt = 'Alicia comprendió al instante que estaba buscando el abanico y el par de guantes blancos de cabritilla, y llena de buena voluntad se puso también ella a buscar por todos lados, pero no encontró ni rastro de ellos. En realidad, todo parecía haber cambiado desde que ella cayó en el charco, y el vestíbulo con la mesa de cristal y la puertecilla habían desaparecido completamente.'
v_dissim_txt = unicode_to_ascii(v_dissim_txt)
maxlen = 300
sim_txt = sim_txt[:maxlen]
dissim_txt = dissim_txt[:maxlen]
v_dissim_txt = v_dissim_txt[:maxlen]

model = decoder

def getPerplexity(txt):
    hidden = model.init_hidden()
    prime_str = ' '
    prime_input = char_tensor(prime_str)
    prob_chain = 0
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for char in txt:
        output, hidden = model(inp, hidden)
        output_dist = output.data.view(-1).exp()
        idx = all_characters.index(char)
        prob = (output_dist/output_dist.sum())[idx].item()
#         print(char, idx, prob)
        prob_chain += np.log(prob)
    perplexity = prob_chain * -1/len(txt)
    return perplexity

print("Perplexity score: %.4f" %getPerplexity(sim_txt))
print("Perplexity score: %.4f" %getPerplexity(dissim_txt))
print("Perplexity score: %.4f" %getPerplexity(v_dissim_txt))