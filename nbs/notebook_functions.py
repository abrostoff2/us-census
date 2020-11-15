import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import product 
from sklearn import preprocessing


def display_group_density_plot(df, groupby, on, palette, figsize):
    """
    Displays a density plot by group, given a continuous variable, and a group to split the data by
    :param df: DataFrame to display data from
    :param groupby: Column name by which plots would be grouped (Categorical, maximum 10 categories)
    :param on: Column name of the different density plots
    :param palette: Color palette to use for drawing
    :param figsize: Figure size
    :return: matplotlib.axes._subplots.AxesSubplot object
    """
    if not isinstance(df, pd.core.frame.DataFrame):
        raise ValueError('df must be a pandas DataFrame')
    if not groupby:
        raise ValueError('groupby parameter must be provided')
    elif not groupby in df.keys():
        raise ValueError(groupby + ' column does not exist in the given DataFrame')
    if not on:
        raise ValueError('on parameter must be provided')
    elif not on in df.keys():
        raise ValueError(on + ' column does not exist in the given DataFrame')
    if len(set(df[groupby])) > 10:
        groups = df[groupby].value_counts().index[:10]
    else:
        groups = set(df[groupby])
    # Get relevant palette
    if palette:
        palette = palette[:len(groups)]
    else:
        palette = sns.color_palette()[:len(groups)]
    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    for value, color in zip(groups, palette):
        sns.kdeplot(df.loc[df[groupby] == value][on], \
                    shade=True, color=color, label=value)
    ax.set_title(str("Distribution of " + on + " per " + groupby + " group"),\
                 fontsize=30)
    ax.set_xlabel(on, fontsize=20)
    return ax


def get_percentage_stats(df, feature):
    return (df[feature].value_counts()/df.shape[0])


def group_lower_ranking_values(df, column):
    rating_counts = df.groupby(column).agg('count')
    pct_value = rating_counts[lambda x: x.columns[0]].quantile(.5)
    values_below_pct_value = rating_counts[lambda x: x.columns[0]].loc[lambda s: s < pct_value].index.values
    def fix_values(row):
        if row[column] in values_below_pct_value:
            row[column] = 'Other'
        return row 
    rating_grouped = df.apply(fix_values, axis=1)
    return rating_grouped



class Process:
    def __init__ (self, data, is_training):
        self.is_training = is_training 
        self.data = data


    def add_headers_and_clean(self):
        with open('/Users/alexbrostoff/us-census/data/census_income_metadata.txt', 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = in_file.read()
        features = re.findall('#\d{1,3} \(([^\)]+)\)', lines)

        for feature in self.data.columns:
            self.data[feature] = self.data[feature].apply(lambda x: x.strip() if type(x)==str else x)

        '''a feature is missing - check from list 
          There is an extra continuous feature
          instance weight'''
        self.data.columns = features[:24] + ['instance weight'] +  features[24:] + ['income']

        #make feature names comply with pep8 standards
        underscore_features =[feature.replace(' ', '_') for feature in self.data.columns]
        self.data.columns=underscore_features
        
        data_types_df = pd.DataFrame(re.findall(r'#\d{1,3} \(([^\)]+)\) (.+)', lines), 
                             columns=['feature', 'feature_type'])
        data_types_df['feature'] = data_types_df['feature'].apply(lambda x: x.replace(' ', '_'))

        self.grouped_data_types_df = data_types_df.groupby(['feature_type', 'feature']).count()
        self.features = features

        self.continuous_features = list(data_types_df[data_types_df['feature_type']=='continuous'].feature)       
        self.nominal_features = list(data_types_df[data_types_df['feature_type']=='nominal'].feature)    
        self.data_types_df = data_types_df  


    def encode_labels(self):
        data = self.data
        le = preprocessing.LabelEncoder()
        le.fit(data['income'])
        data['income'] = le.transform(data['income'])
        self.data = data


    def remove_wealth_features(self):
        self.features_to_drop = ['instance_weight', 'wage_per_hour',
                                                  'capital_gains', 'capital_losses',
                                                  'dividends_from_stocks']
        self.data = self.data.drop(self.features_to_drop, axis=1)
    

    def remove_nulls_and_duplicates(self):
        data = self.data.replace({' ?': np.nan}, inplace=True)
        data = self.data.replace({'?': np.nan}, inplace=True)
        data = self.data.replace({r'(Not in universe).*': np.nan}, regex=True, inplace=True)
        data = self.data.dropna(axis=1, thresh=self.data.shape[0]*.5)
        data.dropna(axis=0, thresh=18, inplace=True)
        self.data = data.drop_duplicates()


    def keep_heads_of_households(self):
        self.data = self.data.loc[self.data.detailed_household_summary_in_household=='Householder']
        self.data[self.data.age>18] #only adults


    def aggregate_hispanic(self):
        self.data['hispanic_origin'] = self.data.hispanic_origin.apply(lambda x: 'Mexican' if x in ('Mexican (Mexicano)', 'Mexican-American', 'Chicano') else x)
        self.data['hispanic_origin'] = self.data.hispanic_origin.apply(lambda x: 'Central or South American' if x in ('Puerto Rican','Cuban') else x)
        self.data['hispanic_origin'] = self.data.hispanic_origin.apply(lambda x: np.nan if x in ('All other','NA', 'Do not know') else x)
        # self.data['race_and_hispanic'] = self.data.apply(lambda x: (x['race'], x['hispanic_origin']), axis=1)


    def make_education_ordinal(self):
        edu_dict = {'Children': 0,
        'Less than 1st grade': 0,
        '1st 2nd 3rd or 4th grade': 0,
        '5th or 6th grade': 0,
        '7th and 8th grade': 1,
        '9th grade': 1,
        '10th grade': 1,
        '11th grade': 1,
        '12th grade no diploma': 1,
        'High school graduate': 2,
        'Some college but no degree':3,
        'Associates degree-occup /vocational':4,
        'Associates degree-academic program': 4,
        'Bachelors degree(BA AB BS)': 5,
        'Masters degree(MA MS MEng MEd MSW MBA)': 6,
        'Prof school degree (MD DDS DVM LLB JD)': 7,
        'Doctorate degree(PhD EdD)': 7}
        self.data['education'] = self.data.education.replace(edu_dict).astype('int')


    def get_country_index_scores(self):
        self.country_data = pd.read_csv('../data/Human development index (HDI).csv', skiprows=[0]).iloc[:,[1,6]].set_index('Country')
        self.data['country_of_birth_self'] = self.data.country_of_birth_self.apply(lambda x: x.replace('-', ' ').strip() if type(x)==str else 0)
        self.data['country_of_birth_mother'] = self.data.country_of_birth_mother.apply(lambda x: x.replace('-', ' ').strip() if type(x)==str else 0)
        self.data['country_of_birth_father'] = self.data.country_of_birth_father.apply(lambda x: x.replace('-', ' ').strip() if type(x)==str else 0)
        self.data = self.data.replace({'Columbia': 'Colombia'})
        self.data.reset_index(inplace=True)
        self.data['self_country'] = pd.merge(self.data, self.country_data, left_on='country_of_birth_self', right_on='Country', how='left')['1994'].apply(lambda x: float(x)).fillna(.5)
        self.data['mother_country'] = pd.merge(self.data, self.country_data, left_on='country_of_birth_mother', right_on='Country', how='left')['1994'].apply(lambda x: float(x)).fillna(.5)
        self.data['father_country'] = pd.merge(self.data, self.country_data, left_on='country_of_birth_father', right_on='Country', how='left')['1994'].apply(lambda x: float(x)).fillna(.5)
        self.data['immigrant_score'] = self.data.apply(lambda x: (x['self_country'] + x['mother_country'] + x['father_country'])/3, axis=1)


    def balance_data(self):
        sampled_data = self.data.sample(frac=1)
        majority_data = sampled_data[sampled_data.income==0].iloc[:self.data.income.value_counts()[1]]
        minority_data = sampled_data[sampled_data.income==1]
        self.data = pd.concat([majority_data, minority_data])


    def fit(self):
        self.add_headers_and_clean()
        self.encode_labels()
        self.remove_wealth_features()
        self.remove_nulls_and_duplicates()
        self.keep_heads_of_households()
        self.aggregate_hispanic()
        self.make_education_ordinal()
        self.get_country_index_scores()
        if self.is_training:
            self.balance_data()


        
