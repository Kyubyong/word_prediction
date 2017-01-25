#/usr/bin/python2
# coding: utf-8
'''
Builds an English news corpus from wikinews.
Feel free to use a prebuilt one such as reuter in nltk if you want,
but it may be too small.

Make sure you download raw wikinews data from 20160701 through 20170101 
at the following links before running this file,
then extract them to `data/raw` folder.

https://dumps.wikimedia.org/enwikinews/20160701/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160720/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160801/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160820/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160901/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20160920/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161001/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161020/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161101/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161120/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161201/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20161220/enwikinews-20170120-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwikinews/20170101/enwikinews-20170120-pages-articles-multistream.xml.bz2
'''
from __future__ import print_function
import numpy as np
import pickle
import codecs
import lxml.etree as ET
import regex
from nltk.tokenize import sent_tokenize

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
    import glob
    
    with codecs.open('data/en_wikinews.txt', 'w', 'utf-8') as fout:
        fs = glob.glob('data/raw/*.xml')
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
                                sents = [regex.sub("([.!?]+$)", r" \1", sent) for sent in sent_tokenize(para.strip())]
                                fout.write(" ".join(sents) + "\n")
                except:
                    continue
                
                elem.clear() # We need to save memory!
                i += 1
                if i % 1000 == 0: print(i,)

if __name__ == '__main__':
    build_corpus()
    print("Done")        