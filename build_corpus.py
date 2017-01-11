#/usr/bin/python2
# coding: utf-8
'''
Build Corpus.
'''
from __future__ import print_function
import numpy as np
import pickle
import re
import codecs
import lxml.etree as ET
import regex
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = regex.sub("\[http[^]]+? ([^]]+)]", r"\1", text) 
    text = regex.sub("\[http[^]]+]", "", text) 
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    text = regex.sub(u"[^ \r\n\p{Latin}\d\-'.?!]", " ", text)
    text = text.lower()
    
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    return text

def build_corpus():
    '''Embeds and vectorize words in corpus'''
    import glob
    
    with codecs.open('data/en_wikinews.txt', 'w', 'utf-8') as fout:
        fs = glob.glob('data/*.xml')
        ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace
        for f in fs:
            i = 1
            for _, elem in ET.iterparse(f, tag=ns+"text"):
                try:
                    if i > 5000:
                        running_text = elem.text
                        running_text = running_text.split("===")[0]
                        running_text = clean_text(running_text)
                        paras = running_text.split("\n")
                        for para in paras:
                            if len(para) > 500:
                                fout.write(" ".join(word_tokenize(para.strip())) + "\n")
                except:
                    continue
                elem.clear() # We need to save memory!
                i += 1
                if i % 1000 == 0: print(i,)

if __name__ == '__main__':
    build_corpus()
    print("Done")        