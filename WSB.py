import pandas as pd
import numpy as np
import datetime as dt
from pytz import timezone
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import emoji
import re
#import spacy
import yfinance as yf
import json
import time
from multiprocessing import Pool

ROOT = '/scratch/network/danieleg/senior-thesis/data'
EMOJIS_INVERSE = inv_map = {v: k for k, v in emoji.UNICODE_EMOJI['en'].items()}
#NLP = spacy.load("en_core_web_sm")
THRESH = 100
STOPS = ['Co','Inc','Ltd','Company','Technologies','ADR','RT','Systems','Industries','Interactive','Holding','Trust',
         'Corporation','Holdings','Energy','Group','Limited','SA','Cor','Acquisition','The','PLC','SPA','Corp','&',
         'Brands','SE','Resorts','Airlines','Cruises','Therapeutics','Pharmaceuticals','Cannabis','Health','Investment',
         'Class','Video','Communications','Companies','AG','Sponsored','International','Net','Technology','Motor',
         'Special','Purpose','Wholesale','Hotels','Pharmaceutical']
EDITS = {'AAL':['American Airlines'],'X':['US Steel','United States Steel'],'CRSR':['Corsair'],
         'AMD':['Advanced Micro Devices'],'SPY':['S&P500','S&P 500','ES','SPX','S&P','S AND P','SP500'],'QQQ':['Nasdaq'],
         'FCX':['Freeport'],'LULU':['Lululemon'],'SLV':['Silver'],'GOOG':['Alphabet','Google'],
         'U':['Unity'],'CLF':['ClevelandCliffs','Cleveland Cliffs'],'TSM':['Taiwan Semiconductor','Taiwan Semi'],
         'AIV':['Apartment Investment'],'AAPL':['Apple','🍏','🍎'],'RKT':['']}

def replace_conames(ser,nicknames_dict):
    for key in tqdm(list(nicknames_dict.keys())):
        ser = ser.str.replace(r'(^|(?<=[^A-Za-z0-9\']))'+key+r'($|(?=[^A-Za-z0-9\']))',nicknames_dict[key],flags=re.IGNORECASE)
    return ser

def replace_cotickers(ser,all_tickers,common_words,ignore):
    for tiq in tqdm(all_tickers):
        if tiq in ignore:
            pass
        elif len(tiq) == 1:
            ser = ser.str.replace(r'(^|(?<=[^A-Za-z\']))\$'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
        elif tiq in common_words:
            ser = ser.str.replace(r'(^|(?<=[^A-Za-z\']))\$'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
            ser = ser.str.replace(r'(^|(?<=[^A-Za-z\'\$]))'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq)
        else:
            ser = ser.str.replace(r'(^|(?<=[^A-Za-z\']))(\$){0,1}'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
    return ser

class WSB:

    def __init__(self,sd=None,ed=None,saved_path=None,saved_df=None,saved_arrivals_path=None,saved_tickers=None,n_cores=1):
        assert((sd!=None and ed!=None) or (type(saved_path)!=type(None)) or (type(saved_df)!=type(None)))
        if type(saved_path)!=type(None):
            self.cleaned = self.load_saved(saved_path)
        elif type(saved_df)!=type(None):
            self.cleaned = saved_df
        else:
            raw = self.load_data(sd,ed)
            self.cleaned = self.find_tickers(raw,n_cores)
        if type(saved_tickers)!=type(None):
            self.tickers = saved_tickers
        else:
            self.tickers = self.get_tickers()
        if type(saved_arrivals_path)!=type(None):
            self.arrivals = np.load(saved_arrivals_path,allow_pickle=True)
        else:
            pass
            #self.arrivals = self.get_arrivals()


    def get_arrivals(self):
        arrivals = []
        for i,ticker in tqdm(enumerate(self.tickers.index)):
            ticker_arrivals = self.find(ticker)
            ticker_arrivals = ticker_arrivals['timestamp'].astype(np.int64).to_numpy()/(10**9)
            arrivals.append(ticker_arrivals)
        return arrivals

    def top_tickers_time(self,sd,ed):

        #start = dt.datetime.strptime(sd,'%Y-%m-%d')
        start = time.mktime(sd.timetuple())
        #end = dt.datetime.strptime(ed,'%Y-%m-%d')
        end = time.mktime(ed.timetuple())
        
        #tickers = list(self.tickers.index.values)
        #ids = [tickers.index('$'+name) for name in names]
        
        len_arrivals = np.array([len(self.arrivals[i][(self.arrivals[i] >= start) & (self.arrivals[i] < end)]) for i in range(len(self.arrivals))])
        return self.tickers.index[np.argsort(len_arrivals)[::-1]]

    def find_arrivals(self,names,sd,ed,delta=60):
        #start = dt.datetime.strptime(sd,'%Y-%m-%d')
        start = time.mktime(sd.timetuple())
        #end = dt.datetime.strptime(ed,'%Y-%m-%d')
        end = time.mktime(ed.timetuple())

        #start = (pd.Series([dt.datetime.strptime(sd,'%Y-%m-%d')]).astype(np.int64) / 10**9)[0]
        #end = (pd.Series([dt.datetime.strptime(ed,'%Y-%m-%d')]).astype(np.int64) / 10**9)[0]
        
        tickers = list(self.tickers.index.values)
        ids = [tickers.index('$'+name) for name in names]
        
        arrivals = [(self.arrivals[i][(self.arrivals[i] >= start) & (self.arrivals[i] < end)]-start)/delta for i in ids]
        return arrivals

    """
    def get_arrivals(self,name,sd,ed,taq,deltas):
        start = timezone('US/Eastern').localize(dt.datetime.strptime(sd,'%Y-%m-%d'))
        end = timezone('US/Eastern').localize(dt.datetime.strptime(ed,'%Y-%m-%d'))
        df = self.find(name)
        
        df = df.loc[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        df['PRICE'] = np.nan
        for delta_mins in deltas:
            col = '{}Min_Chng'.format(delta_mins)
            df[col] = 0
        
        dates = df['timestamp'].dt.date.unique()
        for date in tqdm(dates):
            df_date = df.loc[df['timestamp'].dt.date == date]
            df_date = df_date.reset_index(drop=False).set_index('timestamp').between_time('9:30','16:00',include_end=False)
            df_date = df_date.reset_index(drop=False).set_index('id')
            
            taq_date = taq[['timestamp','MID']].loc[taq['timestamp'].dt.date == date]
            taq_date = taq_date.set_index('timestamp').between_time('9:30','16:00',include_end=False)
            taq_date = taq_date.reset_index()
            taq_date = taq_date.copy()
            taq_date = taq_date.rename(columns={'MID':'PRICE'})
            
            prices = pd.merge_asof(df_date[['timestamp']], taq_date, on='timestamp')
            prices.index = df_date.index
            df.at[df_date.index,'PRICE'] = prices
            
            for delta_mins in deltas:
                col = '{}Min_Chng'.format(delta_mins)
                
                taq_delta = taq_date.copy()
                taq_delta = taq_delta.rename(columns={'PRICE':'{}Min_Chng'.format(delta_mins)})
                taq_delta['timestamp'] += pd.Timedelta(minutes=delta_mins)
                prices = pd.merge_asof(df_date[['timestamp']], taq_delta, on='timestamp')
                prices.index = df_date.index
                #print(prices)
                
                df.at[df_date.index,col] = (df.loc[df_date.index,'PRICE']/prices[col] - 1).fillna(0)
                
        arrivals = df['timestamp'].astype(np.int64).to_numpy()/(10**9) - dt.datetime.timestamp(start)
        arrivals /= 60
        returns = df[['{}Min_Chng'.format(delta_mins) for delta_mins in deltas]].to_numpy()
        return arrivals,returns
    """


    def find_all(self):
        df = self.cleaned[self.cleaned['tickers'].str.len().gt(0) |
                          self.cleaned['reply_tickers'].str.len().gt(0)]
        return df
        
    def find(self,ticker):
        df = self.cleaned[self.cleaned['tickers'].apply(lambda x: ticker in x) |
                          self.cleaned['reply_tickers'].apply(lambda x: ticker in x)]
        return df

    def load_saved(self,saved_path):
        df = []
        for chunk in tqdm(pd.read_csv(saved_path,index_col=0,chunksize=250000,usecols=['id','timestamp','tickers_clean','tickers','reply_tickers'])):
            chunk['tickers'] = chunk['tickers'].str.replace('\'','\"').apply(lambda x: [] if x=='[]' else json.loads(x))
            chunk['reply_tickers'] = chunk['reply_tickers'].str.replace('\'','\"').apply(lambda x: [] if x=='[]' else json.loads(x))
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            df.append(chunk)
        df = pd.concat(df)
        return df

    def load_data(self,sd,ed):
        start = dt.datetime(int(sd[:4]),int(sd[5:7]),int(sd[8:10]))
        end = dt.datetime(int(ed[:4]),int(ed[5:7]),int(ed[8:10]))

        #df = pd.DataFrame()
        df_arr = []
        for folder in [ROOT+'/reddit_DD',ROOT+'/reddit_MT',ROOT+'/reddit_WT',ROOT+'/reddit_JAN']:
            files = os.listdir(folder)
            dates = [dt.datetime(int(file[:4]),int(file[5:7]),int(file[8:10])) for file in files]
            files = [file for i,file in enumerate(files) if (dates[i] >= start and dates[i] < end)]
            for filename in tqdm(files):
                submission = pd.read_csv('{}/{}'.format(folder,filename), dtype={'distinguished': str})
                submission = submission.loc[~submission['body'].isnull()]
                submission.loc[submission['author'].isnull(),'author'] = '[deleted]'
                submission = clean(submission)
                submission['body_clean'] = submission['body'].apply(transform)
                submission = submission.loc[~submission['body_clean'].isnull()]
                df_arr.append(submission)
        df = pd.concat(df_arr)
        df = df.sort_values('timestamp')
        df = df.loc[(df['timestamp'] >= start.astimezone(timezone('US/Eastern'))) & (df['timestamp'] < end.astimezone(timezone('US/Eastern')))]
        return df

    def get_tickers(self,raw=None):
        if type(raw) != type(None):
            tickers = raw['body_clean'].str.findall(r'(?P<ticker>\$[A-Z]+)')
            tickers = np.array(tickers.loc[tickers.str.len().gt(0)].sum())
        else:
            tickers = np.concatenate([np.concatenate(self.cleaned['tickers'].to_numpy()),
                                    np.concatenate(self.cleaned['reply_tickers'].to_numpy())])
            
        values, counts = np.unique(tickers, return_counts=True)
        ind = counts.argsort()
        counts = counts[ind[::-1]]
        values = values[ind[::-1]]
        tickers = pd.Series(counts)
        tickers.index = values
        return tickers

    def top_tickers(self,threshold=THRESH):
        return self.tickers[self.tickers >= threshold]
    
    def find_tickers(self,raw,n_cores):
        self.tickers = self.get_tickers(raw=raw)
        tickers = self.top_tickers()
        tickers_list = [ticker[1:] for ticker in tickers.index]
        dictionary = {}
        company_names = pd.read_csv(ROOT+'/company_names.csv',index_col=0,header=None)
        company_names = company_names.replace(np.nan,'')
        for co in company_names.index:
            dictionary[co] = company_names.loc[co].values[0]
        stops = [stop.lower() for stop in STOPS]
        names = pd.DataFrame(np.array([list(dictionary.keys()),list(dictionary.values())]).T,columns=['symbol','name'])
        names['nickname'] = names['name'].str.replace('\.com','')
        names['nickname'] = names['nickname'].str.replace('[^\w\s\&]','')
        names['nickname'] = names['nickname'].str.split()
        names['nickname'] = names['nickname'].apply(lambda x: x[:3] if (len(x) > 2 and x[1].lower() in ['and','of']) else x[:2])
        names['nickname'] = names['nickname'].apply(lambda x: [wrd for i, wrd in enumerate(x) if not ((wrd.lower() in stops) and (i != 0))])
        names['nickname'] = names['nickname'].apply(lambda x: [wrd for i, wrd in enumerate(x) if not len(wrd)==1])
        names['nickname'] = names['nickname'].apply(lambda x: ' '.join(x))
        names['nickname'].loc[names['symbol'].str.lower()==names['nickname'].str.lower()] = ''
        names['nickname'] = names['nickname'].apply(lambda x: [x])
        
        names['nickname'].loc[names['symbol'].isin(EDITS.keys())] = names['symbol'].loc[names['symbol'].isin(EDITS.keys())].apply(lambda x: EDITS[x])
        nicknames_dict = {}
        for i in range(len(names)):
            symbol = names['symbol'].iloc[i]
            for nickname in names['nickname'].iloc[i]:
                if nickname != '':
                    nicknames_dict[nickname.lower()] = '$'+symbol.upper()
        df = raw.copy()
        df['tickers_clean'] = df['body_clean'].copy()

        def parallelize_dataframe(df, func, n_cores=n_cores):
            df_split = np.array_split(df, n_cores)
            pool = Pool(n_cores)
            df = pd.concat(pool.map(func, df_split))
            pool.close()
            pool.join()
            return df

        df['tickers_clean'] = parallelize_dataframe(df['tickers_clean'],func=lambda x: replace_conames(x,nicknames_dict=nicknames_dict))

        """
        for key in tqdm(list(nicknames_dict.keys())):
            df['tickers_clean'] = df['tickers_clean'].str.replace(r'(^|(?<=[^A-Za-z0-9\']))'+key+r'($|(?=[^A-Za-z0-9\']))',nicknames_dict[key],flags=re.IGNORECASE)
        """

        all_tickers = names['symbol'].values
        common_words = ['DASH','SNOW','NET','EDIT','RIDE','WISH','WORK',
                        'OPEN','SHOP','LOW','COST','SPOT','RUN','EVER','GOLD','BOX','AIR','PLAY']
        ignore = ['MOON','YOLO','IPO','BE']
        df['tickers_clean'] = parallelize_dataframe(df['tickers_clean'],func=lambda x: replace_cotickers(x,all_tickers=all_tickers,common_words=common_words,ignore=ignore))
        """
        all_tickers = names['symbol'].values
        common_words = ['DASH','SNOW','NET','EDIT','RIDE','WISH','WORK',
                         'OPEN','SHOP','LOW','COST','SPOT','RUN','EVER','GOLD','BOX','AIR','PLAY']
        ignore = ['MOON','YOLO','IPO','BE']
        for tiq in tqdm(all_tickers):
            if tiq in ignore:
                pass
            elif len(tiq) == 1:
                df['tickers_clean'] = df['tickers_clean'].str.replace(r'(^|(?<=[^A-Za-z\']))\$'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
            elif tiq in common_words:
                df['tickers_clean'] = df['tickers_clean'].str.replace(r'(^|(?<=[^A-Za-z\']))\$'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
                df['tickers_clean'] = df['tickers_clean'].str.replace(r'(^|(?<=[^A-Za-z\'\$]))'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq)
            else:
                df['tickers_clean'] = df['tickers_clean'].str.replace(r'(^|(?<=[^A-Za-z\']))(\$){0,1}'+tiq+r'($|(?=[^A-Za-z\']))','$'+tiq,flags=re.IGNORECASE)
        """

        df['tickers'] = df['tickers_clean'].str.findall('(\$[A-Z]+)').apply(lambda x: list(set([tick for tick in x if tick[1:] in names['symbol'].values])))
        df['reply_tickers'] = [[] for i in range(len(df))]
        df['reply_tickers'].loc[df['tickers'].str.len().eq(0)] = df['parent_id'].loc[df['tickers'].str.len().eq(0)].apply(lambda idx: df.loc[idx,'tickers'] if idx in df.index else [])
        df['reply_tickers'].loc[df['tickers'].str.len().eq(0) & df['reply_tickers'].str.len().eq(0)] = df['parent_id'].loc[df['tickers'].str.len().eq(0) & df['reply_tickers'].str.len().eq(0)].apply(lambda idx: df.loc[idx,'reply_tickers'] if idx in df.index else [])
        return df
        
def clean(df):
    df = df[['id','created_utc','body','score','author','link_id','parent_id']]
    df = df.rename(columns={'link_id': 'submission_id','created_utc':'timestamp'})
    df = df.set_index('id')
    
    df['submission_id'] = df['submission_id'].apply(lambda x: x[3:])
    df['parent_id'] = df['parent_id'].apply(lambda x: x[3:] if 't1_' in x else np.nan)
    
    df['timestamp'] = df['timestamp'].apply(lambda x: (dt.datetime.fromtimestamp(x)).astimezone(timezone('US/Eastern')))
    df = df.sort_values(by=['timestamp'])
    
    deleted = (df['body'].str.contains('[deleted]',regex=False)) | (df['body'].str.contains('[removed]',regex=False))
    deleted_parents = df.loc[deleted].index
    df = df.loc[~deleted]
    while(True):
        deleted = df['parent_id'].isin(deleted_parents)
        deleted_parents = df.loc[deleted].index
        if deleted_parents.empty:
            break
        df = df.loc[~deleted]
    return df

def transform(comment):
    if type(comment) != str:
        return comment
    new = comment
    
    new = re.sub(r'\:\)', '😊', new, flags=re.IGNORECASE)
    new = re.sub(r'\:\(', '😞', new, flags=re.IGNORECASE)
    new = re.sub(r'\’','\'',new)
    new = re.sub(r'&#x200B;','',new,flags=re.IGNORECASE)
    
    new = [' <emoji>' + emoji.UNICODE_EMOJI['en'][r'{}'.format(letter)][1:-1] + '<emoji> ' if letter in emoji.UNICODE_EMOJI['en'].keys() else letter for letter in new]
    new = ''.join(new)
    new = re.sub(r'[^a-zA-Z0-9\s\n\.\!\?\,\$\-\_\'\<\>\:\/\=\&\%]',' ',new)
    
    new = re.sub(r'(\n| )*\n(\n| )*', '\n', new, flags=re.IGNORECASE)
    new = re.sub(r' +', ' ', new, flags=re.IGNORECASE)
    new = re.sub(r' (?P<punct>(\.|\?|\!|\:)+)',lambda match: ' ' + match.group('punct') + ' ', new, flags=re.IGNORECASE)
    new = re.sub(r'(?<=(\.|\?|\!|\:))(\n)',' ', new, flags=re.IGNORECASE)
    new = re.sub(r'(?!=(\.|\?|\!|\:))(\n)','. ', new, flags=re.IGNORECASE)
    new = re.sub(r'(?P<emoji>(\<emoji\>[^\s]+\<emoji\>))',lambda match: ' ' + EMOJIS_INVERSE[':' + match.group('emoji')[7:-7] + ':'],new)
    new = re.sub(r'[\<\>\_]','',new)
    new = re.sub(r'http[a-zA-Z0-9\.\-\:\/\%\&\=\?]+',' ', new, flags=re.IGNORECASE)
    new = re.sub(r' +', ' ', new, flags=re.IGNORECASE)
    new = new.lstrip().rstrip()
    return new