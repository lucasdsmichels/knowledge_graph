import os
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarWarsQASystem:
    """Sistema de Q&A para o universo Star Wars usando Neo4j + LLM"""
    
    def __init__(self):
        self.setup_environment()
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.graph = self.setup_neo4j_connection()
        self.qa_chain = self.setup_qa_chain()
        
    def setup_environment(self):
        """Configura as variáveis de ambiente necessárias"""
        required_vars = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Variável de ambiente {var} não encontrada")
            os.environ[var] = os.getenv(var)
    
    def setup_neo4j_connection(self) -> Neo4jGraph:
        """Estabelece conexão com Neo4j"""
        try:
            graph = Neo4jGraph(
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
            )
            logger.info("✅ Conexão com Neo4j estabelecida")
            return graph
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com Neo4j: {e}")
            raise
    
    def setup_qa_chain(self) -> GraphCypherQAChain:
        """Configura a chain de Q&A com prompts otimizados"""
        
        # Prompt para geração de Cypher com exemplos
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
Você é um especialista em consultas Cypher para um grafo de conhecimento do universo Star Wars.

Schema do grafo:
{schema}

INSTRUÇÕES IMPORTANTES:
1. Use apenas os nós e relacionamentos disponíveis no schema
2. Sempre use MATCH para encontrar padrões
3. Use WHERE para filtros específicos
4. Limite os resultados com LIMIT quando apropriado
5. Para relacionamentos familiares, use FILHO_DE, IRMÃO_DE, etc.
6. Para afiliações, use PERTENCE_A, ALIADO_DE, etc.

EXEMPLOS:
- "Quem são os filhos de Darth Vader?"
  MATCH (vader:Personagem {{nome: 'Darth Vader'}})-[:FILHO_DE]->(filho:Personagem)
  RETURN filho.nome

- "Quais personagens aparecem no Episódio V?"
  MATCH (p:Personagem)-[:APARECE_EM]->(f:Filme)
  WHERE f.título CONTAINS 'Episódio V'
  RETURN p.nome

- "Quais planetas têm clima árido?"
  MATCH (p:Planeta)
  WHERE p.clima = 'Árido'
  RETURN p.nome, p.tipo

Pergunta: {question}

Gere apenas a consulta Cypher (sem explicações):
"""
        )
        
        # Prompt para resposta final
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Você é um assistente especializado no universo Star Wars. Com base nos dados do grafo de conhecimento, responda à pergunta de forma clara e informativa.

Dados do grafo:
{context}

Pergunta: {question}

INSTRUÇÕES:
1. Responda apenas com base nos dados fornecidos
2. Se não houver dados suficientes, diga claramente
3. Use linguagem natural e seja específico
4. Organize a resposta de forma clara
5. Mencione relacionamentos importantes quando relevante

Resposta:
"""
        )
        
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            include_types=[
                "Personagem", "Filme", "Nave", "Planeta", "Facção", 
                "Espécie", "Droide", "Arma", "Tecnologia", "Evento",
                "Localização", "Organização"
            ],
            verbose=True,
            top_k=10,
            return_intermediate_steps=True
        )
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do grafo"""
        stats = {}
        
        # Contagem de nós por tipo
        node_counts = self.graph.query("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
        """)
        stats['nodes'] = {row['label']: row['count'] for row in node_counts}
        
        # Contagem de relacionamentos por tipo
        rel_counts = self.graph.query("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship, count(r) as count
            ORDER BY count DESC
        """)
        stats['relationships'] = {row['relationship']: row['count'] for row in rel_counts}
        
        return stats
    
    def ask_with_context(self, question: str) -> Dict[str, Any]:
        """Faz pergunta com contexto completo e métricas"""
        try:
            with get_openai_callback() as cb:
                # Executa a pergunta
                response = self.qa_chain.invoke({"query": question})
                
                # Extrai informações intermediárias
                intermediate_steps = response.get("intermediate_steps", [])
                cypher_query = ""
                raw_results = []
                
                if intermediate_steps:
                    for step in intermediate_steps:
                        if isinstance(step, dict):
                            if "query" in step:
                                cypher_query = step["query"]
                            if "context" in step:
                                raw_results = step["context"]
                
                result = {
                    "question": question,
                    "answer": response["result"],
                    "cypher_query": cypher_query,
                    "raw_results": raw_results,
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost
                }
                
                logger.info(f"✅ Pergunta respondida - Tokens: {cb.total_tokens}, Custo: ${cb.total_cost:.4f}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar pergunta: {e}")
            return {
                "question": question,
                "answer": f"Erro ao processar pergunta: {str(e)}",
                "cypher_query": "",
                "raw_results": [],
                "tokens_used": 0,
                "cost": 0.0
            }
    
    def ask_simple(self, question: str) -> str:
        """Faz pergunta simples retornando apenas a resposta"""
        result = self.ask_with_context(question)
        return result["answer"]
    
    def validate_knowledge_base(self) -> Dict[str, Any]:
        """Valida a base de conhecimento com perguntas de teste"""
        test_questions = [
            "Quantos personagens existem no grafo?",
            "Quais são os filmes da trilogia original?",
            "Quais planetas têm clima árido?",
            "Quem são os Jedi no grafo?",
            "Quais eventos acontecem em Tatooine?",
            "Quantos planetas existem no grafo?",
        ]
        
        validation_results = []
        
        for question in test_questions:
            logger.info(f"🧪 Testando: {question}")
            result = self.ask_with_context(question)
            validation_results.append({
                "question": question,
                "has_answer": bool(result["answer"] and "não encontrado" not in result["answer"].lower()),
                "cypher_generated": bool(result["cypher_query"]),
                "answer_length": len(result["answer"]),
                "tokens_used": result["tokens_used"]
            })
        
        # Estatísticas de validação
        total_questions = len(validation_results)
        answered_questions = sum(1 for r in validation_results if r["has_answer"])
        success_rate = (answered_questions / total_questions) * 100
        
        return {
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "success_rate": success_rate,
            "results": validation_results
        }
    
    def demonstrate_anti_hallucination(self) -> Dict[str, Any]:
        """Demonstra como o grafo previne alucinações"""
        
        # Perguntas que podem gerar alucinações sem o grafo
        test_cases = [
            {
                "question": "Qual é a cor do sabre de luz do Mestre Yoda?",
                "expected": "Informação não disponível nos dados"
            },
            {
                "question": "Quantos filhos tem o Imperador Palpatine?",
                "expected": "Informação não disponível nos dados"
            },
            {
                "question": "Qual é o nome da mãe de Luke Skywalker?",
                "expected": "Informação não disponível nos dados"
            }
        ]
        
        results = []
        
        for case in test_cases:
            logger.info(f"🔍 Testando anti-alucinação: {case['question']}")
            result = self.ask_with_context(case["question"])
            
            # Verifica se a resposta indica falta de dados
            indicates_no_data = any(phrase in result["answer"].lower() for phrase in [
                "não encontrado", "não disponível", "não há", "não existe",
                "não foi possível", "sem informações", "não consta"
            ])
            
            results.append({
                "question": case["question"],
                "answer": result["answer"],
                "indicates_no_data": indicates_no_data,
                "prevents_hallucination": indicates_no_data
            })
        
        success_rate = sum(1 for r in results if r["prevents_hallucination"]) / len(results) * 100
        
        return {
            "test_cases": results,
            "anti_hallucination_rate": success_rate
        }

def main():
    """Função principal para demonstração"""
    qa_system = StarWarsQASystem()
    
    print("🎬 SISTEMA DE Q&A STAR WARS")
    print("="*50)
    
    # Mostra estatísticas do grafo
    stats = qa_system.get_graph_stats()
    print(f"📊 Estatísticas do Grafo:")
    print(f"   - Nós: {stats['nodes']}")
    print(f"   - Relacionamentos: {stats['relationships']}")
    print()
    
    # Exemplos de perguntas
    example_questions = [
        "Quais personagens da Facção Rebel Alliance aparecem no Episódio V?",
        "Quem são os filhos de Darth Vader?",
        "Quais planetas têm clima árido?",
        "Quais naves são pilotadas por Han Solo?",
        "Quantos filmes existem no grafo?"
    ]
    
    print("🤖 EXEMPLOS DE PERGUNTAS:")
    print("-" * 30)
    
    for question in example_questions:
        print(f"\n❓ {question}")
        answer = qa_system.ask_simple(question)
        print(f"💬 {answer}")
    
    # Validação da base de conhecimento
    print("\n🧪 VALIDAÇÃO DA BASE DE CONHECIMENTO:")
    print("-" * 40)
    validation = qa_system.validate_knowledge_base()
    print(f"✅ Taxa de sucesso: {validation['success_rate']:.1f}%")
    print(f"📋 Perguntas respondidas: {validation['answered_questions']}/{validation['total_questions']}")
    
    # Demonstração anti-alucinação
    print("\n🛡️ DEMONSTRAÇÃO ANTI-ALUCINAÇÃO:")
    print("-" * 40)
    anti_hallucination = qa_system.demonstrate_anti_hallucination()
    print(f"✅ Taxa de prevenção: {anti_hallucination['anti_hallucination_rate']:.1f}%")
    
    # Interface interativa
    print("\n💬 INTERFACE INTERATIVA:")
    print("-" * 25)
    print("Digite suas perguntas (ou 'quit' para sair):")
    
    while True:
        try:
            question = input("\n❓ Sua pergunta: ").strip()
            if question.lower() in ['quit', 'exit', 'sair']:
                break
            
            if question:
                result = qa_system.ask_with_context(question)
                print(f"💬 {result['answer']}")
                print(f"🔍 Query Cypher: {result['cypher_query']}")
                print(f"💰 Tokens usados: {result['tokens_used']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    print("\n👋 Obrigado por usar o Sistema Q&A Star Wars!")

if __name__ == "__main__":
    main()