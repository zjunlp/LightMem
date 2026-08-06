# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import os
import json
import random
from collections import defaultdict

import datasets
import numpy as np
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook
from src.prompt.base_prompt import AutoPrompt
from src.text_utils.normalize_text import normalize

CLUSTER_NAME2LABELS = {
    "medrxiv": [t.lower() for t in ["Addiction Medicine", "Allergy and Immunology", "Anesthesia", "Cardiovascular Medicine", "Dentistry and Oral Medicine", "Dermatology", "Emergency Medicine",
                    "Endocrinology", "Epidemiology", "Forensic Medicine", "Gastroenterology", "Genetic and Genomic Medicine", "Geriatric Medicine",
                    "Health Economics", "Health Informatics", "Health Policy", "Health Systems and Quality Improvement", "Hematology", "HIV/AIDS", "Infectious Diseases", "Intensive Care and Critical Care Medicine",
                    "Medical Education", "Medical Ethics", "Nephrology", "Neurology", "Nursing", "Nutrition", "Obstetrics and Gynecology", "Occupational and Environmental Health", "Oncology", "Ophthalmology", "Orthopedics", "Otolaryngology",
                    "Pain Medicine", "Palliative Medicine", "Pathology", "Pediatrics", "Pharmacology and Therapeutics", "Primary Care Research", "Psychiatry and Clinical Psychology", "Public and Global Health",
                    "Radiology and Imaging", "Rehabilitation Medicine and Physical Therapy", "Respiratory Medicine", "Rheumatology", "Sexual and Reproductive Health", "Sports Medicine", "Surgery",
                    "Toxicology", "Transplantation", "Urology"]],
    "biorxiv": [t.lower() for t in ["Animal Behavior and Cognition", "Biochemistry", "Bioengineering", "Bioinformatics", "Biophysics", "Cancer Biology", "Cell Biology", "Clinical Trials*", "Developmental Biology",
                    "Ecology", "Epidemiology", "Evolutionary Biology", "Genetics", "Genomics", "Immunology", "Microbiology", "Molecular Biology", "Neuroscience",
                    "Paleontology", "Pathology", "Pharmacology and Toxicology", "Physiology", "Plant Biology", "Scientific Communication and Education", "Synthetic Biology", "Systems Biology", "Zoology"]],
    "arxiv": [
        "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR", "bayes-an",
        "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
        "econ.EM", "econ.GN", "econ.TH", "eess.AS", "eess.IV", "eess.SP", "eess.SY",
        "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG", "math.SP", "math.ST",
        "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con",
        "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI", "nucl-ex", "nucl-th",
        "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus", "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det", "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph", "physics.soc-ph", "physics.space-ph", "quant-ph",
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
        "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
        "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH",
    ],
    "reddit": [
         'amcstock', 'royalfamily', 'FIFA', 'KingkillerChronicle', 'exmuslim', 'EscapefromTarkov', 'CanaryWharfBets',
         'MinecraftServer', 'dirtypenpals', 'hoi4', 'sex', 'ugly', 'Bariloche', 'dpdr', 'covidlonghaulers', 'LOONA',
         'bicycling', 'WhiteWolfRPG', 'DissonautUniverse', 'ABraThatFits', 'lfg', 'Gaming_Headsets', 'GME',
         'Boxing.txt', 'AssassinsCreedValhala', 'LegendsOfRuneterra', '90dayfianceuncensored', 'AskCulinary',
         'DestinyTheGame', 'appliancerepair', 'astrology.txt', 'MonsterHunterWorld', 'TeensMeetTeens', 'Dodocodes',
         'thedivision', 'whowouldwin', 'DMAcademy', 'bicycling.txt', 'Random_Acts_Of_Amazon', 'Sverigesforsvarsmakt',
         'ConnecticutR4R', 'transgendercirclejerk', 'aggretsuko', 'lawschooladmissions', 'SLFmeetups', 'tipofmypenis',
         'cuboulder', 'whatcarshouldIbuy', 'bonnaroo', 'PersonalFinanceCanada', 'podcasts', 'DotA2', 'CreditCards',
         'playstation', 'SummonSign', 'watercooling', 'gardening', 'MLBTheShow', 'askTO', 'ask', 'cryptocoins',
         'Autos.txt', 'MachineLearning', 'exjw', 'NoFeeAC', 'EverMerge', 'McMaster', 'NBA2k', 'Testosterone',
         'KassadinMains', 'Atlanta.txt', 'Stormlight_Archive', 'livesound', 'findaleague', 'Shittyaskflying', 'aorus',
         'VirtualYoutubers', 'arlington', 'oneui', 'overlord', 'xmen', 'MarsWallStreet', 'Wordpress', 'stashinvest',
         'AskGirls', 'idleon', 'depressed', 'AskAnAmerican', 'raisedbynarcissists', 'hitbtc', 'CharacterRant',
         'Cartalk', 'Aerials', 'langrisser', 'comicbooks', 'airsoft.txt', 'MonsterHunter', 'feedthebeast',
         'SaltLakeCity', 'heroesofthestorm', 'HaircareScience', 'AgonGame', 'xboxfindfriends', 'RevenantMain',
         'turtles', 'ChronicPain', 'GameBuilderGarage', 'chicago.txt', 'leagueoflegends', 'AirForce', 'GamingLaptops',
         'HFY', 'AskReddit.txt', 'RedditWritesSeinfeld', 'tipofmytongue', 'unpopularopinion', 'brasil', 'GriefSupport',
         'clonewars', 'ModernMagic', 'LearnerDriverUK', 'silverbugbets', 'atheism.txt', 'vaginismus', 'Screenwriting',
         'BokunoheroFanfiction', 'PokemonTCG', 'mechmarket', 'Doom', 'asktransgender', 'haskell', 'CryptoHorde',
         'pittsburgh', 'teenagersnew', 'FortniteCompetitive', 'gamegrumps', 'BangaloreMains', 'delhi', 'GoodNotes',
         'Anarchism.txt', 'skincareexchange', 'raleigh', 'idlechampions', 'ibs', 'WetlanderHumor', 'AmazonFC',
         'dirtykikpals', 'dbrand', 'ADHD', 'dating_advice', 'AskVet', 'trakstocks', 'graylog', 'teenagers',
         'Bogleheads', 'FiestaST', 'roblox', 'Catholicism', 'nus', 'HilariaBaldwin', 'ELLIPAL_Official', 'CRISPR',
         'conspiracy', 'PanicAttack', 'chess.txt', 'fleshlight', 'Baystreetbets', 'libraryofruina', 'Coins4Sale',
         'InstacartShoppers', 'yoga', 'RocketLeagueExchange', 'appletv', 'paintdotnet', 'GoogleDataStudio',
         'Austin.txt', 'aviation.txt', 'buildapcforme', 'FortNiteBR', 'TMJ', 'sofi', 'popperpigs', 'anime', 'religion',
         'learnmath', 'comedy.txt', 'Buddhism.txt', 'askcarsales', 'brasil.txt', 'GCXRep', 'RoleplayingForReddit',
         'pokemon', 'lasers', 'xmpp', 'nhl', 'OculusQuest', 'GodEater', 'collapse.txt', 'Catholicism.txt', 'BMW',
         'biology.txt', 'blackopscoldwar', 'wicked_edge', 'realonlyfansreviews', 'ModelCars', 'ClashOfClansRecruit',
         'HiveOS', 'candlemagick', 'WallStreetbetsELITE', 'help', 'BlackCountryNewRoad', 'tf2trade', 'redsox',
         'GearsOfWar', 'LifeProTips', 'BuddyCrossing', 'CryptoMarkets', 'Sims4', 'GoRVing', 'JustUnsubbed',
         'WreckingBallMains', 'dragonquest', 'college.txt', 'Kayaking', 'bostonr4r', 'BroadCity', 'selfimprovement',
         'TheMagnusArchives', 'PoonamPandeyFanatics', 'piano', 'JamesHoffmann', 'GunAccessoriesForSale', 'BreakUps',
         'madisonwi', 'controlgame', 'Superstonk', 'KTM', 'Aliexpress', 'HIMYM', 'digimon', 'Crypto_com', 'Bestbuy',
         'cardano', 'software', 'RomanianWolves', 'LeagueConnect', 'kingofqueens', 'NoMansSkyTheGame',
         'hisdarkmaterials', 'sissypersonals', 'premed', 'ShuumatsuNoValkyrie', 'ps3homebrew', 'NoFap', 'Astronomy.txt',
         'Christianity.txt', 'steelseries', 'cancer.txt', 'exalted', 'mfdoom', 'Art.txt', 'emacs', 'spirituality',
         'Coffee.txt', 'CryptoMars', 'miraculousladybug', 'tutanota', 'Animals.txt', 'boston.txt', 'gaming',
         'SatoshiStreetBets', 'seoul', 'SakuraGakuin', 'Sat', 'apple.txt', 'touhou', 'Gta5Modding', 'KikRoleplay',
         'igcse', 'cryptostreetbets', 'amazon.txt', 'COVID19positive', 'Superhero_Ideas', 'ladybusiness',
         'starcraft2coop', 'salesforce', 'HauntingOfHillHouse', 'scatstories', 'belgium.txt', 'TherosDMs',
         'doordash_drivers', 'MarylandUnemployment', 'quittingkratom', 'weather', 'ABA', 'FridayNightFunkin',
         'AskLosAngeles', 'mac', 'coins.txt', 'Bible', 'Reduction', 'vcu', 'samsung', 'relationship_advice', 'NiceHash',
         'theta_network', 'Minecraft', 'HypnoFair', 'Market76', 'photography', 'onewheel', 'kdramarecommends', 'turo',
         'destiny2', 'Throwers', 'extrarfl', 'Kengan_Ashura', 'islam', 'AmItheAsshole', 'AmongUs', 'logh', 'AFL.txt',
         'Scotch', 'RepTime', 'Hedgehog', 'Unemployment', 'alcoholicsanonymous', 'u_mawadom118', 'cars.txt', 'oculus',
         'nanocurrency', 'Tomorrowland', 'newtothenavy', 'gorillaz', 'amateurradio.txt', 'MondoGore', 'kansascity',
         'AstroGaming', 'BDSMcommunity', 'masseffect', 'BitLifeApp', 'boardgames', 'australia.txt', 'visualization',
         'foodscience', 'comicswap', 'capitalism_in_decay', 'FanFiction', 'anime.txt', 'buildapc', 'FitnessDE',
         'baltimore.txt', 'suggestmeabook', 'brave_browser', 'IRS', 'dogecoin', 'canada.txt', 'blender.txt', 'KGBTR',
         'China.txt', 'classicalmusic.txt', 'ClashOfClans', 'AirReps', 'Julia', 'apexlegends', 'MindHunter',
         'books.txt', 'HotWheels', 'puppy101', 'deadbydaylight', 'GlobalPowers', 'FireEmblemHeroes', 'LightNovels',
         'DnD', 'SauceSharingCommunity', 'screenplaychallenge', 'dreamsmp', 'obs', 'cats.txt', 'thomasthetankengine',
         'tarot', 'CPTSD', 'aromantic', 'architecture.txt', 'texas', '40kLore', 'thetagang', 'cscareerquestionsEU',
         'beer.txt', 'KazuhaMains', 'Library', 'MtF', 'wireshark', 'breakingmom', 'MatthiasSubmissions', 'seduction',
         'JoJolion', 'Eldenring', 'leafs', 'chemistry.txt', 'socialanxiety', 'namenerds', 'ShieldAndroidTV', 'Bath',
         'SquaredCircle', 'MECoOp', 'emotestories', 'ACTrade', 'stocks', 'lockpicking', 'detrans', 'NameThatSong',
         'amex', 'gonewildaudio', 'Huel', 'PokemonGoFriends', 'eu4', 'sexstories', 'UKJobs', 'IBO', 'NoContract',
         'accesscontrol', 'stalker', 'TwoXChromosomes', 'Midsommar', 'Blogging.txt', 'rpg', 'transpositive', 'NTU',
         'Wallstreetsilver', 'ZombsRoyale', 'generationology', 'whatsthatbook', 'ConquerorsBlade', 'BenignExistence',
         '196', 'qBittorrent', 'Dyson_Sphere_Program', 'PerkByDaylight', 'Prolactinoma', 'steinsgate',
         'VinylCollectors', 'survivinginfidelity', 'techsupport', 'Vent', 'AskGames', 'TooAfraidToAsk',
         'ValorantBrasil', 'offmychest', 'NassauCountyHookups', 'endocrinology', 'Evernote', 'Netherlands',
         'comicbooks.txt', 'ITCareerQuestions', 'fantasywriters', 'GameSale', 'HyperRP', 'snuffrp', 'Smallville',
         'advertising.txt', 'PokemonHome', 'bleach', 'CultoftheFranklin', 'Paladins', 'xbox', 'Wavyhair', 'dating',
         'Advice.txt', 'SofiawithanF', 'GayYoungOldDating', 'self', 'tomorrow', 'metalgearsolid', 'depression',
         'aliens', 'feminineboys', 'MensRights', 'facebook', 'EliteDangerous', 'SluttyConfessions', 'auntienetwork',
         'passive_income', '1200isplentyketo', 'imaginarypenpals', 'choiceofgames', 'bravelydefault',
         'FiddlesticksMains', 'CozyGrove', 'ApplyingToCollege', 'chronotrigger', 'Adopted', 'askseddit', 'Guitar',
         'Nioh', 'KillingEve', 'Revu', 'C25K', 'youtubers', 'STAYC'
    ],
    "stackexchange": [
        'ell.stackexchange.com.txt', 'crafts.stackexchange.com.txt', 'ebooks.stackexchange.com.txt', 'chinese.stackexchange.com.txt', 'biology.stackexchange.com.txt', 'aviation.stackexchange.com.txt', 'economics.stackexchange.com.txt', 'diy.stackexchange.com.txt', 'craftcms.stackexchange.com.txt', 'apple.stackexchange.com.txt', 'christianity.stackexchange.com.txt', 'civicrm.stackexchange.com.txt', 'askubuntu.com.txt',
        'cseducators.stackexchange.com.txt', 'chemistry.stackexchange.com.txt', 'android.stackexchange.com.txt', 'conlang.stackexchange.com.txt', 'codegolf.stackexchange.com.txt', 'bitcoin.stackexchange.com.txt', 'elementaryos.stackexchange.com.txt', 'devops.stackexchange.com.txt', 'blender.stackexchange.com.txt', 'cstheory.stackexchange.com.txt', 'electronics.stackexchange.com.txt', 'academia.stackexchange.com.txt',
        'emacs.stackexchange.com.txt', 'bicycles.stackexchange.com.txt', 'dsp.stackexchange.com.txt', 'chess.stackexchange.com.txt', 'ai.stackexchange.com.txt', 'cooking.stackexchange.com.txt', 'astronomy.stackexchange.com.txt', 'coffee.stackexchange.com.txt', 'beer.stackexchange.com.txt', 'drupal.stackexchange.com.txt', 'datascience.stackexchange.com.txt', 'crypto.stackexchange.com.txt', 'cs.stackexchange.com.txt',
        'avp.stackexchange.com.txt', 'boardgames.stackexchange.com.txt', 'earthscience.stackexchange.com.txt', 'codereview.stackexchange.com.txt', 'arduino.stackexchange.com.txt', 'anime.stackexchange.com.txt', 'bioinformatics.stackexchange.com.txt', 'dba.stackexchange.com.txt', 'bricks.stackexchange.com.txt', 'cogsci.stackexchange.com.txt', 'buddhism.stackexchange.com.txt', 'computergraphics.stackexchange.com.txt'
    ],
    "news_category": [
        'ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE', 'COMEDY', 'CRIME', 'CULTURE & ARTS', 'DIVORCE',
        'EDUCATION', 'ENTERTAINMENT', 'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK', 'GOOD NEWS', 'GREEN',
        'HEALTHY LIVING', 'HOME & LIVING', 'IMPACT', 'LATINO VOICES', 'MEDIA', 'MONEY',
        'PARENTING', 'PARENTS', 'POLITICS', 'QUEER VOICES', 'RELIGION', 'SCIENCE', 'SPORTS', 'STYLE', 'STYLE & BEAUTY',
        'TASTE', 'TECH', 'THE WORLDPOST', 'TRAVEL', 'U.S. NEWS',
        'WEDDINGS', 'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS', 'WORLDPOST'
    ],
    "20newsgroups": ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
         'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
         'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
         'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
}

@add_metainfo_hook
def data_prepare(examples, dataset_name, data_format,
                 query_prompt='', doc_prompt='', num_hardneg=0,
                 **kwargs):
    '''
    data_type=against_text: sample multiple texts of different labels as negatives
    data_type=against_label: sample multiple labels as negatives
    '''
    contexts, queries, pos_docs, neg_docs = [], [], [], []
    data_type = kwargs.get("data_type", "default")  # against_text or against_label

    ex = json.loads(examples['text'][0])
    title_field_name = [l for l in ["title", "headline"] if l in ex][0]
    abs_field_name = [l for l in ["abstract", "short_description"] if l in ex]
    abs_field_name = abs_field_name[0] if abs_field_name else None
    label_field_name = [l for l in ["category", "categories", "label"] if l in ex][0]
    label_count = defaultdict(int)

    # against_text mode, we gather examples by labels
    label2texts, label2negtexts = defaultdict(list), defaultdict(list)
    if data_type == "against_text":
        for data_idx, raw_text in enumerate(examples['text']):
            example = json.loads(raw_text)
            title = example[title_field_name]
            abstract = example[abs_field_name] if abs_field_name else ""
            pos_label = example[label_field_name]
            text = title if data_format == "s2s" else title + "\n" + abstract
            pos_label = pos_label.split()[0] if label_field_name == "categories" else pos_label
            label2texts[pos_label].append(text)
            label_count[pos_label] += 1
        for pos, texts in label2texts.items():
            for neg_l in label2texts.keys():
                if neg_l != pos:
                    label2negtexts[neg_l].extend(texts)
        for pos in label2texts.keys():
            # since later we only sample a slice, so we shuffle the whole list first
            random.shuffle(label2negtexts[pos])
        # for pos, count in sorted(label_count.items(), key=lambda k:k[1], reverse=True):
        #     print(pos, count)

    # labels for sampling negatives
    label_set = CLUSTER_NAME2LABELS[dataset_name]
    for data_idx, raw_text in enumerate(examples['text']):
        try:
            example = json.loads(raw_text)
            title = example[title_field_name]
            abstract = example[abs_field_name] if abs_field_name else ""
            pos_label = example[label_field_name]
            text = title if data_format == "s2s" else title + "\n" + abstract
            # arxiv has multiple labels
            pos_labels = pos_label.split() if label_field_name == "categories" else [pos_label]
            query = normalize(query_prompt + text)
            if data_type == "against_text":
                if len(label2texts[pos_labels[0]]) == 1:
                    # skip if only one text in that category is available (query/pos are the same)
                    continue
                pos = random.choice(label2texts[pos_labels[0]])
                pos = normalize(query_prompt + pos)
            else:
                pos_label = np.random.choice(pos_labels, size=1, replace=False)[0]
                if "arxiv" in dataset_name and random.random() < 0.5 and ("." in pos_label or "-" in pos_label):  # randomly only use the top-level category, e.g. cs.AI -> cs
                    pos_label = pos_label.replace("-", ".").split(".")[0]
                pos = normalize(doc_prompt + pos_label)
            # add NEGs
            if num_hardneg > 0:
                if data_type == "against_text":
                    # sample negs from texts of other labels
                    neg_texts = label2negtexts[pos_labels[0]]
                    # use a random slice to reduce the sampling time
                    _start = random.randint(0, max(0, len(neg_texts) - num_hardneg * 2))
                    _neg_texts = neg_texts[_start: _start+num_hardneg * 2]
                    if len(_neg_texts) == 0:
                        # in extreme cases neg_texts can be empty (all data in the batch belong to the same category), we use a truncated text as neg
                        _neg_texts = [text[len(text) // 2: ]]
                    negs = np.random.choice(_neg_texts, size=num_hardneg, replace=len(_neg_texts) < num_hardneg)
                    negs = [normalize(query_prompt + neg) for neg in negs]
                else:
                    negs = [l for l in label_set if l not in pos_labels]
                    negs = np.random.choice(negs, size=num_hardneg, replace=len(negs) < num_hardneg)
                    negs = [normalize(doc_prompt + neg) for neg in negs]
                neg_docs.append(negs)
            queries.append(query)
            pos_docs.append(pos)
            contexts.append(raw_text)
        except Exception as e:
            print(f'Error in processing {dataset_name} data, id={id}')
            raise e
    # return_dict = {'contexts': contexts, 'queries': queries, 'docs': pos_docs}
    # if num_hardneg > 0:
    #     return_dict['neg_docs'] = neg_docs
    #     return_dict['neg_contexts'] = contexts
    batch_len = len(queries)
    return_dict =  {"query_text": queries, "query_image": [None]*batch_len,
                   "pos_text": pos_docs, "pos_image": [None]*batch_len,
                   "neg_text": neg_docs if neg_docs else [None]*batch_len, "neg_image": [None]*batch_len}
    return return_dict

DATASET_PARSER_NAME = "mteb_cluster"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mteb_cluster(model_args, data_args, training_args,
                      dataset_name, file_path=None,
                      query_prompt_type='e5mistral', doc_prompt_type='e5mistral', *args, **kwargs):
    assert os.path.isfile(file_path), f'{file_path} does not exist.'

    prompt_name = kwargs.pop("prompt_name") if "prompt_name" in kwargs else data_args["prompt_name"]
    data_format = kwargs.pop("data_format") if "data_format" in kwargs else data_args["data_format"]

    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    kwargs['model_backbone'] = model_args.model_backbone
    query_prompt, doc_prompt = "", ""
    if query_prompt_type:
        query_prompt = AutoPrompt.instantiate(prompt_family=query_prompt_type, task_name=prompt_name, **kwargs)["q_prompt"]
    if doc_prompt_type:
        doc_prompt = AutoPrompt.instantiate(prompt_family=doc_prompt_type, task_name=prompt_name, **kwargs)["d_prompt"]

    assert data_format in ["p2p", "s2s"]
    dataset = datasets.load_dataset("text", split="train", data_files=file_path, keep_in_memory=False, streaming=False)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    remove_columns = ["text"]
    # len(20newsgroups)=5926, batch size has to be smaller than each data size due to bugs in datasets>=2.19.2
    if "20newsgroups" in dataset_name:
        batch_size = 1024 * 4
    # elif "reddit" in dataset_name or "stackexchange" in dataset_name:
    #     batch_size = 1024 * 256
    else:
        batch_size = 1024*8
    num_hardneg = kwargs.get("num_hardneg", data_args.num_hardneg)
    dataset = dataset.map(lambda x: data_prepare(x, num_hardneg=num_hardneg,
                                                 dataset_name=dataset_name, data_format=data_format,
                                                 query_prompt=query_prompt, doc_prompt=doc_prompt,
                                                 **kwargs),
                          drop_last_batch=True,
                          batched=True, batch_size=batch_size,
                          remove_columns=remove_columns)
    # this will add dataset.features, it is supposed to be called at hf_dataset_fns L188,
    #   but at there it may returns nothing (time out?) and leads to being ignored in the training, so we call it here beforehand
    # dataset = dataset._resolve_features()
    # dataset = dataset.shuffle(buffer_size=1024*16, seed=configs.seed)
    # num_rows in iterable_dataset is overridden, set it here for printing dataset stats
    setattr(dataset, 'num_rows', num_rows)

    return dataset
