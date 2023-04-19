import pandas as pd 
import streamlit as st 
import seaborn as sns 
from pycaret.regression import load_model, predict_model 

modelo1 = load_model('meu-modelo-para-charges')
modelo2 = load_model('meu-modelo-para-smoker')

st.sidebar.header('**Streamlit Deploy**') 
 

opcoes = ['Boas-vindas', 
		  'Dashboard',
		  'Custos de Seguro', 
		  'Probabilidade de Fraude']

pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)
 
def smap(x):  
				y = 'male' if x == 'Masculino' else 'female' 
				return y

def rmap(x):
	if x == 'Sudeste':
		return 'southeast'
	elif x == 'Noroeste':
		return 'northwest'
	elif x == 'Sudoeste':
		return 'southwest' 
	else:
		return 'northeast'

def fmap(x):  
	y = 'yes' if x == 'Sim' else 'no' 
	return y

def classificador(modelo, dados):
	pred = predict_model(estimator = modelo, data = dados) 
	return pred

###### PAGINA INICIAL ######

if pagina == 'Boas-vindas': 
  
	st.write("""
	# Prof. Dr. Ricardo Rocha

	Olá! Muito prazer. Meu nome é Ricardo, atualmente sou docente do magistério superior na Universidade Federal da Bahia.  

	Atuo na área de Estatística Computacional junto ao [Departamento de Estatística](https://est.ufba.br/) do 
	Instituto de Matemática e Estatística da UFBA.  

	Além disso, sou o atual coordenador do [Laboratório de Estatística e Data Science (IME-UFBA)](http://led.ufba.br/), 
	onde trabalhamos em projetos com alunos, promovemos eventos e marcamos presença online. 

	O propósito dessa página é funcionar como um Currículo Iterativo. 
	Através de **Streamlit + Markdown**, podemos fazer páginas muito rapidamente, além de ter toda versatilidade do Python rodando a aplicação no back-end.

	Pretendo incluir diversas features nessa página, das quais:

	### Funcionalidades no momento
	- :heavy_check_mark:  Página de boas-vindas 
	- :heavy_check_mark:  Página de Produção Científica
	- :black_square_button:  Página de Resumo de Aulas
	- :black_square_button: Página de Atividade de Extensão
	- :heavy_check_mark:  Página de Projetos Interativos 


	Fique a vontade para entrar em contato, você pode utilizar qualquer uma das redes sociais abaixo!
 
	""")
		 
	col1, col2, col3, col4, col5 = st.columns(5)
 
	col1.markdown('[![](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/linkedin.png)](https://www.linkedin.com/in/ricardorocha86/)')  
	col2.markdown('[![](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/github.png)](https://github.com/ricardorocha86/)')
	col3.markdown('[![](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/instagram.png)](https://www.instagram.com/ricardorocha23/)')
	col4.markdown('[![](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/twitter.png)](https://twitter.com/ricardorocha_86)')
	col5.markdown('[![](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/gmail.png)](mailto:ricardo8610@gmail.com)')



if pagina == 'Dashboard': 
	url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
	dados = pd.read_csv(url)

	col1, col2, col3 = st.columns(3)
	st.markdown('---')

	regiao = col1.selectbox("Região em que mora", dados['region'].unique()) 
	sexo = col2.selectbox("Sexo", ['Masculino', 'Feminino']) 
	criancas = col3.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])


	#fumante = 'yes' if fumante == 'Sim' else 'no'
	sexo = 'male' if sexo == 'Masculino' else 'female'

	filtro_regiao = dados['region'] == regiao 
	filtro_sexo = dados['sex'] == sexo
	filtro_criancas = dados['children'] == criancas
	
	filtro_dados = dados.loc[filtro_regiao & filtro_sexo & filtro_criancas] 

	col1, col2 = st.columns([1,3])

	col1.metric('Idade Média', round(filtro_dados['age'].mean(), 1))
	col1.metric('IMC Médio', round(filtro_dados['bmi'].mean(), 1))
	col1.metric('Custos Médio', round(filtro_dados['charges'].mean(), 1))
	col1.metric('Fumantes', '{:.2%}'.format(filtro_dados['smoker'].value_counts(normalize = True)['yes']))
	
	fig = sns.scatterplot(data = filtro_dados, x = 'bmi', y = 'charges', hue = 'smoker')
	col2.pyplot(fig.get_figure() )

	st.markdown('---')




if pagina == 'Custos de Seguro':
	#st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide2.JPG?raw=true)')

	st.markdown('# Modelagem de valor do seguro')

	st.markdown('Nessa seção é feito o deploy do modelo para cotar o valor do seguro para um indivíduo.\
			Entre com os dados e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	col1, col2, col3= st.columns(3)

	idade = col1.number_input('Idade', 18, 65, 30)
	sexo = col1.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = col2.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = col2.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])
	fumante = col3.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = col3.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])
 

	dados_dicio = {'age': [idade], 
				   'sex': [smap(sexo)], 
				   'bmi': [imc], 
				   'children': [criancas], 
				   'region': [rmap(regiao)], 
				   'smoker': [fmap(fumante)]}
		
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'): 

		saida = classificador(modelo1, dados)
		pred = float(saida['prediction_label'].round(2))  

		s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred) 
 
		st.markdown('## **' + s1 + '**')  






###### PAGINA: MODELO DE FRAUDE ######	


if pagina == 'Probabilidade de Fraude':
	#st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide3.JPG?raw=true)')

	st.markdown('# Detectar probabilidade de fraude')

	st.markdown('Nessa seção é feito o deploy do modelo para detectar probabilidade de fraude na \
		     variável "fumante". Entre com os dados do indivíduo\
		      em análise e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	col1, col2, col3= st.columns(3)

	idade = col1.number_input('Idade', 18, 65, 30)
	sexo = col1.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = col2.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = col2.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5]) 
	regiao = col3.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

	custos = col3.number_input('Custos da pessoa', 1000, 64000, 10000)
 
	dados_dicio = {'age': [idade], 
				   'sex': [smap(sexo)], 
				   'bmi': [imc], 
				   'children': [criancas], 
				   'region': [rmap(regiao)], 
				   'charges': [custos]}
		
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo2, dados) 
		resp = 'NÃO' if saida['prediction_label'][0] == 'no' else 'SIM' 
		prob = saida['prediction_score'][0]  
		s = '{}, com propensão {:.2f}%.'.format(resp, 100*prob)
		st.markdown('## **' + s + '**') 
 