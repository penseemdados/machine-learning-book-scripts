# Estudo de Caso 1 - Engenharia de Atributos com Variáveis Categóricas na Prática

# Definindo o diretório de trabalho
# Este código define a pasta onde seus arquivos estão localizados.
# Descomente e ajuste o caminho conforme necessário.
# setwd("~/projetos/livro01/machine-learning-book-scripts/EstudoCaso01")

# Verifica o diretório de trabalho atual.
# getwd()

# Introdução:
# Modelos de aprendizado de máquina têm dificuldade em interpretar dados categóricos.
# A engenharia de atributos nos permite recontextualizar esses dados para melhorar o 
# desempenho dos modelos de Machine Learning. Além disso, adiciona camadas de perspectiva
# na análise de dados.

# A grande questão que a engenharia de atributos responde é: como podemos utilizar nossos
# dados de maneiras interessantes e inteligentes para torná-los mais úteis?

# Importante:
# Engenharia de atributos não é sobre limpeza de dados ou remoção de valores nulos
# (isso é Data Wrangling); trata-se de alterar variáveis para melhorar a informação
# que elas proporcionam.

# Vamos explorar alguns exemplos práticos de engenharia de atributos usando um dataset
# com dados bancários de usuários.

# Dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# Carregando os dados
# Carrega o dataset a partir de um arquivo CSV.
dataset_bank <- read.table("bank/bank-full.csv", header = TRUE, sep = ";")

# Exibe o dataset em uma visualização de tabela.
View(dataset_bank)

# Exemplo 1 - Criação de Nova Coluna

# Problema: Dados categóricos como preditores podem ter níveis com poucas ocorrências
# ou serem redundantes.
# Solução: Agrupar níveis estrategicamente. Primeiro, usamos a função table() para ver
# a distribuição dos níveis.

# Conta a frequência de cada nível na coluna "job".
table(dataset_bank$job)

# Visualizando a distribuição dos níveis com um gráfico de barras
# Carrega os pacotes dplyr e ggplot2 para manipulação de dados e criação de gráficos.
library(dplyr)
library(ggplot2)

# Agrupa os dados pelo nível de "job", conta a frequência e cria um gráfico de barras.
dataset_bank %>%
  group_by(job) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = job, y = n)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Agora, vamos categorizar as profissões (job) de acordo com o uso da tecnologia:
# baixo, médio ou alto.
# Usamos a função mutate do dplyr para criar a nova coluna.

# Cria uma nova coluna "technology_use" com base na coluna "job".
dataset_bank <- dataset_bank %>%
  mutate(technology_use = case_when(
    job == 'admin.' ~ "medio",
    job == 'blue-collar' ~ "baixo",
    job == 'entrepreneur' ~ "alto",
    job == 'housemaid' ~ "baixo",
    job == 'management' ~ "medio",
    job == 'retired' ~ "baixo",
    job == 'self-employed' ~ "baixo",
    job == 'services' ~ "medio",
    job == 'student' ~ "alto",
    job == 'technician' ~ "alto",
    job == 'unemployed' ~ "baixo",
    job == 'unknown' ~ "baixo"
  ))

# Exibe o dataset atualizado em uma visualização de tabela.
View(dataset_bank)

# Revisamos a nova coluna para verificar a distribuição dos níveis
# Conta a frequência de cada nível na nova coluna "technology_use".
table(dataset_bank$technology_use)

# Calculando a proporção de cada nível
# Calcula a proporção de cada nível na nova coluna "technology_use" e arredonda para 2 casas decimais.
round(prop.table(table(dataset_bank$technology_use)), 2)

# Exemplo 2 - Variáveis Dummies

# Problema: A coluna 'default' indica se um usuário entrou no cheque especial
# com níveis "yes" e "no".
# Solução: Codificamos como uma variável dummy, onde "yes" vira 1 e "no" vira 0.

# Cria uma nova coluna "defaulted" que contém 1 se "default" for "yes" e 0 se for "no".
dataset_bank <- dataset_bank %>%    
  mutate(defaulted = ifelse(default == "yes", 1, 0))

# Exibe o dataset atualizado em uma visualização de tabela.
View(dataset_bank)

# Exemplo 3 - One-Hot Encoding

# Problema: Codificação de variáveis categóricas com muitos níveis.
# Solução: Usamos a codificação one-hot, criando colunas binárias para cada nível
# da variável.

# Carrega o pacote caret para manipulação de dados.
library(caret)

# Criando variáveis dummies para todas as colunas
# Cria variáveis dummies para todas as colunas categóricas.
dmy <- dummyVars(" ~ .", data = dataset_bank)
bank.dummies <- data.frame(predict(dmy, newdata = dataset_bank))

# Visualizando a nova tabela com as variáveis dummies
# Exibe a nova tabela com as variáveis dummies.
View(bank.dummies)
str(bank.dummies)

# Exemplo 4 - Combinando Recursos

# Problema: A combinação de variáveis pode melhorar o desempenho preditivo.
# Solução: Agrupamos duas variáveis e contamos as ocorrências.

# Agrupa os dados pelas colunas "job" e "marital" e conta a frequência de cada combinação.
dataset_bank %>%
  group_by(job, marital) %>%
  summarise(n = n())

# Visualizando a combinação de variáveis com um gráfico de barras
# Cria um gráfico de barras para visualizar a combinação das variáveis "job" e "marital".
dataset_bank %>%
  group_by(job, marital) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = job, y = n, fill = marital)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Criando variáveis dummies para a combinação de job e marital
# Cria variáveis dummies para a combinação de "job" e "marital".
dmy <- dummyVars( ~ job:marital, data = dataset_bank)
bank.cross <- predict(dmy, newdata = dataset_bank)

# Exibe a nova tabela com as variáveis dummies para a combinação de "job" e "marital".
View(bank.cross)

# Nota: Ao combinar variáveis, verifique a esparsidade dos novos valores e aplique 
# técnicas apropriadas conforme necessário.

# Conclusão
# Existem muitos outros métodos para realizar engenharia de atributos com variáveis 
# numéricas e combinações de variáveis categóricas e numéricas. Podemos usar PCA, entre 
# outras técnicas, para melhorar o poder preditivo das variáveis explicativas.
