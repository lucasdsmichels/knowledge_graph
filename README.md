## Alunos
  - Lucas Michels
  - Gustavo Larsen
  - Thiago de Freitas Saraiva
  - Joao Antonio David
  - Luis Felipe Mondini

## Tema e Motivação

Este projeto consiste na construção de um **Grafo de Conhecimento** do universo **Star Wars**, integrado a um modelo de linguagem (LLM) para permitir consultas em linguagem natural sobre entidades e relacionamentos do universo. A motivação principal é demonstrar como um grafo bem estruturado pode servir como uma base de conhecimento confiável, reduzindo alucinações e melhorando a precisão das respostas geradas pela LLM.

## Modelo de Grafo Adotado

O grafo é composto por:

* **Nós (Labels)**

  * `Personagem`, `Filme`, `Planeta`, `Nave`, `Facção`, `Espécie`, `Droide`, `Arma`, `Tecnologia`, `Evento`, `Localização`, `Organização`

* **Relacionamentos (Types)**

  * `APARECE_EM`, `PILOTA`, `PERTENCE_A`, `TEM_CENA_EM`, `É_CONTROLADO_POR`, `NASCEU_EM`, `VIVE_EM`, `LUTA_CONTRA`, `ALIADO_DE`, `FILHO_DE`, `MESTRE_DE`, `POSSUI`, `LOCALIZADO_EM`, `ACONTECE_EM`, `PARTICIPA_DE`, `CRIADO_POR`, `USADO_POR`

* **Propriedades de nós**

  * Exemplos: `nome`, `título`, `ano`, `diretor`, `clima`, `população`, `espécie`, `planeta_natal`, `afiliação`, `lado_da_força`

* **Propriedades de relacionamentos**

  * Exemplos: `tipo`, `duração`, `importância`, `período`, `resultado`

Além disso, são criadas **restrições** e **índices** para garantir unicidade (`nome`, `título`) e melhorar a performance de consulta.

## Estratégia de Importação e Fontes de Dados

1. **Carga de Documentos PDF**:

   * Utiliza `PyPDFLoader` para carregar arquivos `.pdf` contendo informações sobre o universo Star Wars.
   * Os documentos são divididos em chunks de até 1000 caracteres com sobreposição de 150, para otimizar a extração de entidades.

2. **Transformação em Grafos**:

   * `LLMGraphTransformer` extrai nós e relacionamentos de cada chunk, respeitando o modelo de grafo definido.
   * Os dados extraídos são inseridos diretamente no Neo4j.

3. **Conhecimento Manual**:

   * Operações `MERGE` para inserir manualmente filmes da trilogia original, personagens principais, planetas e relacionamentos familiares, garantindo informações essenciais mesmo quando não presentes nos PDFs.

4. **Validação e Estatísticas**:

   * Contagem de nós e relacionamentos para verificar a completude do grafo.
   * Perguntas de teste para medir taxa de sucesso e prevenir alucinações.

## Exemplos de Perguntas que a LLM Pode Responder

A LLM, em conjunto com o grafo, permite responder consultas como:

* **Quais personagens da Facção Rebel Alliance aparecem no Episódio V?**
* **Quem são os filhos de Darth Vader?**
* **Quais planetas têm clima árido?**
* **Quais naves são pilotadas por Han Solo?**
* **Quantos filmes existem no grafo?**

Essas perguntas são convertidas em consultas Cypher pelo `GraphCypherQAChain` antes de serem executadas no Neo4j, e a resposta final é gerada em linguagem natural pela LLM.

## Vídeo
https://github.com/user-attachments/assets/bacdf06c-ede7-4772-803b-bc479cd7553a


