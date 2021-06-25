from glob import glob
from argparse import ArgumentParser
import spacy
import os, json, sys
from diaparser.parsers import Parser


def get_spacy_model(lang):
    if lang == 'eng':
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    elif lang == 'deu':
        os.system("python -m spacy download de_core_news_sm")
        nlp = spacy.load("de_core_news_sm")
    elif lang == 'fra':
        os.system("python -m spacy download fr_core_news_sm")
        nlp = spacy.load("fr_core_news_sm")
    elif lang == 'nld':
        os.system("python -m spacy download nl_core_news_sm")
        nlp = spacy.load("nl_core_news_sm")
    elif lang == 'por':
        os.system("python -m spacy download pt_core_news_sm")
        nlp = spacy.load("pt_core_news_sm")
    elif lang == 'rus':
        os.system("python -m spacy download ru_core_news_sm")
        nlp = spacy.load("ru_core_news_sm")
    elif lang == 'spa':
        os.system("python -m spacy download es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")
    elif lang == 'zho':
        os.system("python -m spacy download zh_core_web_sm")
        nlp = spacy.load("zh_core_web_sm")
    else:
        # eus and tur
        os.system("python -m spacy download xx_ent_wiki_sm")
        nlp = spacy.load("xx_ent_wiki_sm")

    return nlp


def get_diaparser_model(lang, model_dir):
    if lang == 'eng':
        parser = Parser.load('en_ewt-electra')
    elif lang == 'deu':
        parser = Parser.load('de_hdt.dbmdz-bert-base')
    elif lang == 'fra':
        parser = Parser.load('fr_sequoia.camembert')
    elif lang == 'nld':
        parser = Parser.load('nl_alpino_lassysmall.wietsedv')
    elif lang == 'rus':
        parser = Parser.load('ru_syntagrus.DeepPavlov')
    elif lang == 'spa':
        parser = Parser.load('es_ancora.mbert')
    elif lang == 'zho':
        parser = Parser.load('zh_ptb.hfl')
    elif lang == 'tur':
        parser = Parser.load('tr_boun.electra-base')
    elif lang == 'por':
        parser = Parser.load(model_dir + '/diaparser.pt_bosque.bert-base-portuguese-cased.pt')
    elif lang == 'eus':
        parser = Parser.load(model_dir + '/diaparser.eu_bdt.distilbert-multilingual-cased.pt')

    return parser


def get_tags(sentences, lang):
    nlp = get_spacy_model(lang)
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    tg = {'lemma': [], 'pos1': [], 'pos2': []}
    for doc in nlp.pipe(sentences):
        tg['lemma'].append([])
        tg['pos1'].append([])
        tg['pos2'].append([])
        for token in doc:
            tg['lemma'][-1].append(token.lemma_)
            tg['pos1'][-1].append(token.pos_)
            tg['pos2'][-1].append(token.tag_)

    return tg


def dependency_parser(sentences, lang, model_dir):
    parser = get_diaparser_model(lang, model_dir)
    dataset = parser.predict(sentences, prob=True)
    return dataset


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-f", "--file", help="dir containing subdirectories for each languages, including sent_test.pred "
                                        "file",
                   default='/Users/shabnam/Desktop/GU/Projects/DISRPT/sharedtask2019/correctDATA/data/2019-output'
                           '-testing/')
    p.add_argument("-i", "--inf", help="dir containing doc info, output of get_docs_inof.py",
                   default='/Users/shabnam/Desktop/GU/Projects/DISRPT/sharedtask2019/correctDATA/data/2019-output'
                           '-testing/')
    p.add_argument("-d", "--model_dir", help="directory containing diaparser.eu_bdt.distilbert-multilingual-cased.pt "
                                             "and diaparser.pt_bosque.bert-base-portuguese-cased.pt for diaparser.",
                   default="/Users/shabnam/Desktop/GU/Projects/DISRPT/pretrained/")
    p.add_argument("-m", "--mode", help="train/test/dev.",
                   default="dev")
    opts = p.parse_args()

    folders = glob(opts.file + '*/')
    for data_dir in folders:
        with open(data_dir + '/sent_' + opts.mode + '.txt', 'r') as inp:
            lang = data_dir.split('/')[-2].split('.')[0]
            sentences = []
            for line in inp:
                if line.startswith('<s>'):
                    sentences.append([])
                elif line.startswith('</s>'):
                    continue
                else:
                    sentences[-1].append(line.rstrip())
            tags = get_tags(sentences, lang)
            data = dependency_parser(sentences, lang, opts.model_dir)
            with open(data_dir + 'docs_tokens' + opts.mode + '.json') as f:
                inf = json.load(f)

            with open(data_dir + '/' + data_dir.split('/')[-2] + '_' + opts.mode + '_silver.conll', 'r') as ot:
                start = 0
                ot.write('# newdoc id = ' + inf['docs'][0] + '\n')
                tok_index = 0
                doc_index = 0
                for i in range(start, len(sentences)):
                    lns = data[i].split('\n')
                    for j in range(len(sentences[i])):
                        if tok_index == len(inf['toks'][doc_index]):
                            ot.write('# newdoc id = ' + inf['docs'][doc_index + 1] + '\n')
                            tok_index = 0
                            doc_index += 1
                        ann = lns[j].split('\t')
                        if ann[1] != inf['toks'][doc_index][tok_index]:
                            print("tokens not matching")
                            sys.exit()
                        res = ann[0] + '\t' + ann[1] + '\t' + tags['lemma'][i][j] + '\t' + tags['pos1'][i][
                            j] + '\t' + tags['pos2'][i][j] + '\t' + ann[5] + '\t' + ann[6] + '\t' + ann[7] + '\t' + \
                              ann[8] + '\t' + ann[8] + '\n'
                        ot.write(res)
