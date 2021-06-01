#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from scipy import stats


# In[21]:


acoes_varejo = ['MGLU3.SA', 'VVAR3.SA', 'BTOW3.SA']


# In[22]:


acoes_varejo_df = pd.DataFrame()
for acao in acoes_varejo:
    acoes_varejo_df[acao] = data.DataReader(acao, data_source='yahoo', start='2015-01-01')['Close']
    


# In[23]:


acoes_varejo_df.isnull().sum()


# In[24]:


acoes_varejo_df.shape


# In[25]:


acoes_varejo_df.to_csv('acoes_varejo.csv')


# In[26]:


acoes_varejo_df = pd.read_csv('acoes_varejo.csv')


# In[27]:


acoes_varejo_df


# In[29]:


acoes_varejo_df.columns[1:]


# In[31]:


acoes_varejo_df.describe()


# In[33]:


sns.histplot(acoes_varejo_df['BTOW3.SA']);


# In[34]:


acoes_varejo_df = acoes_varejo_df.rename(columns={'MGLU3.SA': 'MGLU', 'VVAR3.SA': 'VVAR', 'BTOW3.SA': 'BTOW'})


# In[35]:


plt.figure(figsize=(10,50))
i = 1
for i in np.arange(1, len(acoes_varejo_df.columns)):
    plt.subplot(7, 1, i + 1)
    sns.histplot(acoes_varejo_df[acoes_varejo_df.columns[i]], kde = True)
    plt.title(acoes_varejo_df.columns[i])


# In[36]:


sns.boxplot(x = acoes_varejo_df['BTOW'])

plt.figure(figsize=(10,50))
i = 1
for i in np.arange(1, len(acoes_varejo_df.columns)):
    plt.subplot(7, 1, i + 1)
    sns.boxplot(acoes_varejo_df[acoes_varejo_df.columns[i]])
    plt.title(acoes_varejo_df.columns[i])


# In[37]:


acoes_varejo_df.plot(x = 'Date', figsize = (15,7), title = 'Histórico das Ações')


# In[38]:


acoes_varejo_df_normalizado = acoes_varejo_df.copy() # salvar uma cópia do registro original
for i in acoes_varejo_df_normalizado.columns[1:]: 
    acoes_varejo_df_normalizado[i] = acoes_varejo_df_normalizado[i] / acoes_varejo_df_normalizado[i][0] #dividir uma linha, sempre pela primeira linha


# In[39]:


acoes_varejo_df_normalizado


# In[40]:


figura = px.line(title = 'Histórico das Ações')
for i in acoes_varejo_df.columns[1:]:
    figura.add_scatter(x = acoes_varejo_df['Date'], y = acoes_varejo_df[i], name=i)
figura.show()  


# In[41]:


figura = px.line(title = 'Histórico das Ações - Normalizado')
for i in acoes_varejo_df_normalizado.columns[1:]:
    figura.add_scatter(x = acoes_varejo_df_normalizado['Date'], y = acoes_varejo_df_normalizado[i], name=i)
figura.show()   


# In[45]:


dataset = pd.read_csv('acoes_varejo.csv')


# In[46]:


dataset = dataset.rename(columns={'MGLU3.SA': 'MGLU', 'VVAR3.SA': 'VVAR', 'BTOW3.SA': 'BTOW'})


# In[95]:


dataset


# In[96]:


#TAXA DE RETORNO SIMPLES DAS AÇÕES - ANUAL
dataset['RS MGLU'] = (dataset['MGLU'] / dataset['MGLU'].shift(1)) - 1
dataset['RS VVAR'] = (dataset['VVAR'] / dataset['VVAR'].shift(1)) - 1
dataset['RS BTOW'] = (dataset['BTOW'] / dataset['BTOW'].shift(1)) - 1

dataset['RS BTOW'].plot();


# In[97]:


#MÉDIA DA TAXA DE RETORNO SIMPLES (ANUAL) - FORMA MANUAL
(dataset['RS MGLU'].mean() * 246) * 100 # O ano comercial tem 246 dias


# In[98]:


(dataset['RS VVAR'].mean() * 246) * 100 # O ano comercial tem 246 dias


# In[99]:


(dataset['RS BTOW'].mean() * 246) * 100 # O ano comercial tem 246 dias


# In[100]:


np.log(dataset['MGLU'][len(dataset) - 1] / dataset['MGLU'][0]) * 100


# In[101]:


np.log(dataset['VVAR'][len(dataset) - 1] / dataset['VVAR'][0]) * 100


# In[102]:


np.log(dataset['BTOW'][len(dataset) - 1] / dataset['BTOW'][0]) * 100


# In[103]:


dataset.drop(labels = ['RS MGLU', 'RS VVAR', 'RS BTOW'], axis = 1, inplace = True)


# In[104]:


dataset # Confirmar se está sem a coluna DATE


# In[109]:


taxas_retorno = (dataset / dataset.shift(1)) - 1


# In[110]:


taxas_retorno


# In[106]:


#RISCO SISTEMÁTICO

pesos = np.array([0.33,0.33,0.33])


# In[107]:


pesos.sum()


# In[108]:


#VARIÂNCIA ANUAL - o quão distante os valores estão da média
taxas_retorno.var() * 246


# In[111]:


variancia_pesos = (taxas_retorno.var() * 246 * pesos)


# In[113]:


variancia_pesos


# In[116]:


variancia_portfolio = np.dot(pesos, np.dot(taxas_retorno.cov() * 246, pesos))


# In[118]:


variancia_portfolio


# In[119]:


sub = variancia_pesos[0] - variancia_pesos[1] - variancia_pesos[2]


# In[120]:


sub


# In[121]:


# RISCO NÃO SISTEMATICO PARA A CARTEIRA COM TODAS AS AÇÕES
risco_nao_sistematico = (variancia_portfolio - sub)


# In[122]:


risco_nao_sistematico


# In[124]:


# Carregar o arquivo inicial para o Cálculo dos retornos anuais

dataset1 = pd.read_csv('acoes_varejo.csv')


# In[126]:


dataset1 = dataset1.rename(columns={'MGLU3.SA': 'MGLU', 'VVAR3.SA': 'VVAR', 'BTOW3.SA': 'BTOW'})


# In[127]:


dataset1


# # Cálculo do Retorno Anual - 2015

# In[161]:


dataset1['MGLU'][dataset1['Date'] =='2015-01-02'], dataset1['MGLU'][dataset1['Date'] =='2015-12-30']
np.log(0.06 / 0.23) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[162]:


dataset1['VVAR'][dataset1['Date'] =='2015-01-02'], dataset1['VVAR'][dataset1['Date'] =='2015-12-30']
np.log(1.5 / 6.8) * 100


# In[163]:


dataset1['BTOW'][dataset1['Date'] =='2015-01-02'], dataset1['BTOW'][dataset1['Date'] =='2015-12-30']
np.log(14.70 / 21.14) * 100


# # Cálcuo do Retorno Anual - 2016

# In[164]:


dataset1['MGLU'][dataset1['Date'] =='2016-01-04'], dataset1['MGLU'][dataset1['Date'] =='2016-12-29']
np.log(0.41 / 0.07) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[165]:


dataset1['VVAR'][dataset1['Date'] =='2016-01-04'], dataset1['VVAR'][dataset1['Date'] =='2016-12-29']
np.log(3.6 / 1.2) * 100


# In[166]:


dataset1['BTOW'][dataset1['Date'] =='2016-01-04'], dataset1['BTOW'][dataset1['Date'] =='2016-12-29']
np.log(9.84 / 13.98) * 100


# # Cálculo do Retorno Anual - 2017

# In[167]:


dataset1['MGLU'][dataset1['Date'] =='2017-01-02'], dataset1['MGLU'][dataset1['Date'] =='2017-12-29']
np.log(2.50/ 0.39) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[168]:


dataset1['VVAR'][dataset1['Date'] =='2017-01-02'], dataset1['VVAR'][dataset1['Date'] =='2017-12-29']
np.log(7.93/ 3.8) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[169]:


dataset1['BTOW'][dataset1['Date'] =='2017-01-02'], dataset1['BTOW'][dataset1['Date'] =='2017-12-29']
np.log(20.26/ 9.84) * 100 #TAXA DE RETORNO LOGARÍTMICA


# # Cálculo do Retorno Anual - 2018

# In[170]:


dataset1['MGLU'][dataset1['Date'] =='2018-01-02'], dataset1['MGLU'][dataset1['Date'] =='2018-12-28']
np.log(5.65/ 2.47) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[171]:


dataset1['VVAR'][dataset1['Date'] =='2018-01-02'], dataset1['VVAR'][dataset1['Date'] =='2018-12-28']
np.log(4.39/ 7.71) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[172]:


dataset1['BTOW'][dataset1['Date'] =='2018-01-02'], dataset1['BTOW'][dataset1['Date'] =='2018-12-28']
np.log(41.53/ 19.76) * 100 #TAXA DE RETORNO LOGARÍTMICA


# # Cálculo do Retorno Anual - 2019

# In[173]:


dataset1['MGLU'][dataset1['Date'] =='2019-01-02'], dataset1['MGLU'][dataset1['Date'] =='2019-12-30']
np.log(11.92/ 5.81) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[174]:


dataset1['VVAR'][dataset1['Date'] =='2019-01-02'], dataset1['VVAR'][dataset1['Date'] =='2019-12-30']
np.log(11.17/ 4.38) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[175]:


dataset1['BTOW'][dataset1['Date'] =='2019-01-02'], dataset1['BTOW'][dataset1['Date'] =='2019-12-30']
np.log(62.86/ 42.50) * 100 #TAXA DE RETORNO LOGARÍTMICA


# # Cálculo do Retorno Anual - 2020

# In[176]:


dataset1['MGLU'][dataset1['Date'] =='2020-01-02'], dataset1['MGLU'][dataset1['Date'] =='2020-12-30']
np.log(24.95/ 12.33) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[177]:


dataset1['VVAR'][dataset1['Date'] =='2020-01-02'], dataset1['VVAR'][dataset1['Date'] =='2020-12-30']
np.log(16.16/ 11.73) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[178]:


dataset1['BTOW'][dataset1['Date'] =='2020-01-02'], dataset1['BTOW'][dataset1['Date'] =='2020-12-30']
np.log(75.61/ 65.18) * 100 #TAXA DE RETORNO LOGARÍTMICA


# # Cálculo do Retorno Anual - 2021

# In[179]:


dataset1['MGLU'][dataset1['Date'] =='2021-01-04'], dataset1['MGLU'][dataset1['Date'] =='2021-05-21']
np.log(18.53/ 25.20) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[180]:


dataset1['VVAR'][dataset1['Date'] =='2021-01-04'], dataset1['VVAR'][dataset1['Date'] =='2021-05-21']
np.log(11.87/ 16.17) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[181]:


dataset1['BTOW'][dataset1['Date'] =='2021-01-04'], dataset1['BTOW'][dataset1['Date'] =='2021-05-21']
np.log(56/ 75.18) * 100 #TAXA DE RETORNO LOGARÍTMICA


# In[195]:


# VARIÂNCIA ANUAL - MGLU
taxas_mglu = np.array([-134.37, 176.76, 185.78, 82.74, 70.48, -30.74])
taxas_mglu.var()


# In[197]:


# VARIÂNCIA ANUAL - VVAR
taxas_vvar = np.array([-151.14, 109.86, 73.56, -56.31, 93.61, 32.03, -30.91])
taxas_vvar.var()


# In[198]:


taxas_btow = np.array([-36.33, 35.11, 72.21, 74.27, 39.14, 14.84, -29.45])
taxas_btow.var()


# In[202]:


#DESVIO PADRÃO - MGLU
desvio_padrao_mglu = math.sqrt(taxas_mglu.std()) 
desvio_padrao_mglu


# In[203]:


#DESVIO PADRÃO - VVAR
desvio_padrao_vvar = math.sqrt(taxas_vvar.std()) 
desvio_padrao_vvar  


# In[204]:


#DESVIO PADRÃO - BTOW
desvio_padrao_btow = math.sqrt(taxas_btow.std()) 
desvio_padrao_btow  


# In[205]:


#COEFICIENTE DE VARIAÇÃO - MGLU
stats.variation(taxas_mglu) * 100


# In[206]:


#COEFICIENTE DE VARIAÇÃO - VVAR
stats.variation(taxas_vvar) * 100


# In[207]:


#COEFICIENTE DE VARIAÇÃO - BTOW
stats.variation(taxas_btow) * 100


# In[208]:


# Cálculo do Risco médio anual
dataset1.drop(labels = ['Date'], axis = 1, inplace = True)


# In[209]:


taxas_retorno = (dataset / dataset.shift(1)) - 1

taxas_retorno.std() * 100


# In[210]:


# Por ano
taxas_retorno.std() * 246


# In[211]:


# risco das ações anualizado
taxas_retorno.std() * math.sqrt(246)


# In[212]:


#COVARIÂNCIA
taxas_retorno.cov()


# In[214]:


#COEFICIENTE DE CORRELAÇÃO
taxas_retorno.corr()


# In[216]:


plt.figure(figsize=(8,8))
sns.heatmap(taxas_retorno.corr(),annot = True);


# In[217]:


#RISCO DE UM PORTIFÓLIO COM DUAS AÇÕES###
taxas_retorno_mglu_btow = taxas_retorno.drop(columns = ['VVAR'])


# In[219]:


taxas_retorno_mglu_btow.cov()


# In[221]:


taxas_retorno_mglu_btow.cov() * 246


# In[223]:


pesos1 = np.array([0.5,0.5])
np.dot(taxas_retorno_mglu_btow.cov() * 246, pesos1)


# In[224]:


np.dot(pesos1, np.dot(taxas_retorno_mglu_btow.cov() * 246, pesos1))


# In[225]:


#DESVIO PADRÃO - SIGNIFICA A VOLATILIDADE DA CARTEIRA
math.sqrt(np.dot(pesos1, np.dot(taxas_retorno_mglu_btow.cov() * 246, pesos1)))


# In[226]:


#RISCO DE UM PORTIFÓLIO COM TODAS AÇÕES##

pesos2 = np.array([0.33,0.33,0.33])


# In[227]:


#VALOR DE COVARIÂNCIA ANUAL
taxas_retorno.cov() * 246


# In[228]:


np.dot(taxas_retorno.cov() * 246, pesos2)


# In[230]:


variancia_portfolio = np.dot(pesos2, np.dot(taxas_retorno.cov() * 246, pesos2))
variancia_portfolio


# In[231]:


volatilidade_portfolio = math.sqrt(variancia_portfolio)
volatilidade_portfolio


# # Alocação e Otimização de Portifólio

# In[278]:


dataset2 = pd.read_csv('acoes_varejo.csv')


# In[279]:


dataset2


# In[280]:


dataset2 = dataset2.rename(columns={'MGLU3.SA': 'MGLU', 'VVAR3.SA': 'VVAR', 'BTOW3.SA': 'BTOW'})


# In[281]:


dataset2


# In[283]:


#Alocação Aleatória de Ativos
def alocacao_ativos(dataset2, dinheiro_total, seed = 0, melhores_pesos = []):
  dataset2 = dataset2.copy()

  if seed != 0:
    np.random.seed(seed)

  if len(melhores_pesos) > 0:
    pesos2 = melhores_pesos
  else:  
    pesos2 = np.random.random(len(dataset2.columns) - 1)
    #print(pesos, pesos.sum())
    pesos2 = pesos2 / pesos2.sum()
    #print(pesos, pesos.sum())

  colunas = dataset2.columns[1:]
  #print(colunas)
  for i in colunas:
    dataset2[i] = (dataset2[i] / dataset2[i][0])

  for i, acao in enumerate(dataset2.columns[1:]):
    #print(i, acao)
    dataset2[acao] = dataset2[acao] * pesos2[i] * dinheiro_total
  
  dataset2['soma valor'] = dataset2.sum(axis = 1)

  datas = dataset2['Date']
  #print(datas)

  dataset2.drop(labels = ['Date'], axis = 1, inplace = True)
  dataset2['taxa retorno'] = 0.0

  for i in range(1, len(dataset)):
    dataset2['taxa retorno'][i] = ((dataset2['soma valor'][i] / dataset2['soma valor'][i - 1]) - 1) * 100

  acoes_pesos = pd.DataFrame(data = {'Ações': colunas, 'Pesos': pesos2 * 100})

  return dataset2, datas, acoes_pesos, dataset2.loc[len(dataset2) - 1]['soma valor']

dataset2, datas, acoes_pesos, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, 10)

dataset2

datas

acoes_pesos

soma_valor


# In[284]:


dataset2


# # Análise Gráfica

# In[287]:


figura1 = px.line(x = datas, y = dataset2['taxa retorno'], title = 'Retorno diário do portifólio')
figura1.show()


# In[292]:


figura2 = px.line(title = 'Evolução do patrimônio')
for i in dataset2.drop(columns = ['soma valor', 'taxa retorno']).columns:
  figura2.add_scatter(x = datas, y = dataset2[i], name = i)
figura2.show()


# In[291]:


figura3 = px.line(x = datas, y = dataset2['soma valor'], title = 'Evolução do patrimônio')
figura3.show()


# In[293]:


# Retorno acumulado em todo o período
dataset2.loc[len(dataset2) - 1]['soma valor'] / dataset2.loc[0]['soma valor'] - 1


# In[294]:


dataset2['taxa retorno'].std()


# In[295]:


# Sharpe ratio

(dataset2['taxa retorno'].mean() / dataset2['taxa retorno'].std()) * np.sqrt(246)
dinheiro_total = 5000
soma_valor - dinheiro_total


# In[296]:


taxa_selic_2015 = 12.75
taxa_selic_2016 = 14.25
taxa_selic_2017 = 12.25
taxa_selic_2018 = 6.50
taxa_selic_2019 = 5.0
taxa_selic_2020 = 2.0
taxa_selic_2021 = 5.5


# In[298]:


valor_2015 = dinheiro_total + (dinheiro_total * taxa_selic_2015 / 100)
valor_2015


# In[299]:


valor_2016 = valor_2015 + (valor_2015 * taxa_selic_2016 / 100)
valor_2016


# In[300]:


valor_2017 = valor_2016 + (valor_2016 * taxa_selic_2017 / 100)
valor_2017


# In[301]:


valor_2018 = valor_2017 + (valor_2017 * taxa_selic_2018 / 100)
valor_2018


# In[302]:


valor_2019 = valor_2018 + (valor_2018 * taxa_selic_2019 / 100)
valor_2019


# In[303]:


valor_2020 = valor_2019 + (valor_2019 * taxa_selic_2020 / 100)
valor_2020


# In[304]:


rendimentos = valor_2020 - dinheiro_total
rendimentos


# In[305]:


ir = rendimentos * 15 / 100
ir


# In[306]:


valor_2020 - ir


# In[307]:


taxa_selic_historico = np.array([12.75, 14.25, 12.25, 6.5, 5.0, 2.0, 5.5])
taxa_selic_historico.mean() / 100


# In[308]:


(dataset2['taxa retorno'].mean() - taxa_selic_historico.mean() / 100) / dataset2['taxa retorno'].std() * np.sqrt(246)


# # Otimização de portfólio - randômico - Markowitz

# In[309]:


import sys
1 - sys.maxsize


# In[327]:


dataset2 = pd.read_csv('acoes_varejo.csv')
dataset2 = dataset2.rename(columns={'MGLU3.SA': 'MGLU', 'VVAR3.SA': 'VVAR', 'BTOW3.SA': 'BTOW'})


# In[328]:


dataset2


# In[331]:


def alocacao_portfolio(dataset2, dinheiro_total, sem_risco, repeticoes):
  dataset2 = dataset2.copy()
  dataset_original = dataset2.copy()

  lista_retorno_esperado = []
  lista_volatilidade_esperada = []
  lista_sharpe_ratio = []

  melhor_sharpe_ratio = 1 - sys.maxsize
  melhores_pesos = np.empty
  melhor_volatilidade = 0
  melhor_retorno = 0
  
  for _ in range(repeticoes):
    pesos4 = np.random.random(len(dataset2.columns) - 1)
    pesos4 = pesos4 / pesos4.sum()

    for i in dataset2.columns[1:]:
      dataset2[i] = dataset2[i] / dataset2[i][0]

    for i, acao in enumerate(dataset2.columns[1:]):
      dataset2[acao] = dataset2[acao] * pesos4[i] * dinheiro_total

    dataset2.drop(labels = ['Date'], axis = 1, inplace=True)

    retorno_carteira = np.log(dataset2 / dataset2.shift(1))
    matriz_covariancia = retorno_carteira.cov()

    dataset2['soma valor'] = dataset2.sum(axis = 1)
    dataset2['taxa retorno'] = 0.0

    for i in range(1, len(dataset2)):
      dataset2['taxa retorno'][i] = np.log(dataset2['soma valor'][i] / dataset2['soma valor'][i - 1])

    #sharpe_ratio = (dataset['taxa retorno'].mean() - sem_risco) / dataset['taxa retorno'].std() * np.sqrt(246)
    retorno_esperado = np.sum(dataset2['taxa retorno'].mean() * pesos4) * 246
    volatilidade_esperada = np.sqrt(np.dot(pesos4, np.dot(matriz_covariancia * 246, pesos4)))
    sharpe_ratio = (retorno_esperado - sem_risco) / volatilidade_esperada

    if sharpe_ratio > melhor_sharpe_ratio:
      melhor_sharpe_ratio = sharpe_ratio
      melhores_pesos = pesos
      melhor_volatilidade = volatilidade_esperada
      melhor_retorno = retorno_esperado

    lista_retorno_esperado.append(retorno_esperado)
    lista_volatilidade_esperada.append(volatilidade_esperada)
    lista_sharpe_ratio.append(sharpe_ratio)
    
    dataset2 = dataset_original.copy()

  return melhor_sharpe_ratio, melhores_pesos, lista_retorno_esperado, lista_volatilidade_esperada, lista_sharpe_ratio, melhor_volatilidade, melhor_retorno

sharpe_ratio, melhores_pesos, ls_retorno, ls_volatilidade, ls_sharpe_ratio, melhor_volatilidade, melhor_retorno = alocacao_portfolio(pd.read_csv('acoes_varejo.csv'), 5000, taxa_selic_historico.mean() / 100, 1000)


# In[332]:


sharpe_ratio, melhores_pesos


# In[334]:


_, _, acoes_pesos, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, melhores_pesos=melhores_pesos)


# In[335]:


acoes_pesos, soma_valor

print(ls_retorno)

print(ls_volatilidade)

print(ls_sharpe_ratio)

melhor_retorno, melhor_volatilidade


# In[336]:


plt.figure(figsize=(10,8))
plt.scatter(ls_volatilidade, ls_retorno, c = ls_sharpe_ratio)
plt.colorbar(label = 'Sharpe ratio')
plt.xlabel('Volatilidade')
plt.ylabel('Retorno')
plt.scatter(melhor_volatilidade, melhor_retorno, c = 'red', s = 100);


# In[337]:


get_ipython().system('pip install mlrose')


# In[342]:


import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose


# In[350]:


taxa_selic_historico

dataset_original = pd.read_csv('acoes_varejo.csv')
dinheiro_total = 5000
sem_risco = taxa_selic_historico.mean() / 100


# In[351]:


def fitness_function(solucao):
  dataset3 = dataset_original.copy()
  pesos5 = solucao / solucao.sum()

  for i in dataset3.columns[1:]:
    dataset3[i] = (dataset3[i] / dataset3[i][0])

  for i, acao in enumerate(dataset3.columns[1:]):
    dataset3[acao] = dataset3[acao] * pesos5[i] * dinheiro_total

  dataset3.drop(labels = ['Date'], axis = 1, inplace=True)
  dataset3['soma valor'] = dataset3.sum(axis = 1)
  dataset3['taxa retorno'] = 0.0

  for i in range(1, len(dataset3)):
    dataset3['taxa retorno'][i] = ((dataset3['soma valor'][i] / dataset3['soma valor'][i - 1]) - 1) * 100

  sharpe_ratio = (dataset3['taxa retorno'].mean() - sem_risco) / dataset3['taxa retorno'].std() * np.sqrt(246)

  return sharpe_ratio


# In[352]:


np.random.seed(10)
pesos5 = np.random.random(len(dataset_original.columns) - 1)
pesos5 = pesos5 / pesos5.sum()
pesos5

fitness_function(pesos5)


# In[353]:


def visualiza_alocacao(solucao):
  colunas = dataset_original.columns[1:]
  for i in range(len(solucao)):
    print(colunas[i], solucao[i] * 100)


# In[354]:


visualiza_alocacao(pesos5)


# In[355]:


fitness = mlrose.CustomFitness(fitness_function)


# In[361]:


problema_maximizacao = mlrose.ContinuousOpt(length=3, fitness_fn=fitness,
                                            maximize = True, min_val = 0, max_val = 1)


# In[362]:


problema_minimizacao = mlrose.ContinuousOpt(length=3, fitness_fn=fitness,
                                            maximize = False, min_val = 0, max_val = 1)


# # Hill climb

# In[363]:


melhor_solucao, melhor_custo = mlrose.hill_climb(problema_maximizacao, random_state = 1)
melhor_solucao, melhor_custo


# In[364]:


melhor_solucao = melhor_solucao / melhor_solucao.sum()
melhor_solucao, melhor_solucao.sum()


# In[365]:


visualiza_alocacao(melhor_solucao)


# In[366]:


_, _, _, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, melhores_pesos=melhor_solucao)
soma_valor


# In[367]:


pior_solucao, pior_custo = mlrose.hill_climb(problema_minimizacao, random_state = 1)
pior_solucao, pior_custo


# In[368]:


pior_solucao = pior_solucao / pior_solucao.sum()
pior_solucao, pior_solucao.sum()


# In[369]:


visualiza_alocacao(pior_solucao)


# In[370]:


_, _, _, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, melhores_pesos=pior_solucao)
soma_valor


# # Simulated annealing

# In[371]:


melhor_solucao, melhor_custo = mlrose.simulated_annealing(problema_maximizacao, random_state = 1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
melhor_solucao, melhor_custo


# In[372]:


visualiza_alocacao(melhor_solucao)


# In[373]:


_, _, _, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, melhores_pesos=melhor_solucao)
soma_valor


# # Algoritmo genético

# In[374]:


problema_maximizacao_ag = mlrose.ContinuousOpt(length = 3, fitness_fn = fitness, 
                                               maximize = True, min_val = 0.1, max_val = 1)


# In[375]:


melhor_solucao, melhor_custo = mlrose.genetic_alg(problema_maximizacao_ag, random_state = 1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
melhor_solucao, melhor_custo


# In[376]:


visualiza_alocacao(melhor_solucao)


# In[377]:


_, _, _, soma_valor = alocacao_ativos(pd.read_csv('acoes_varejo.csv'), 5000, melhores_pesos=melhor_solucao)
soma_valor


# In[ ]:




