import os
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarWarsKnowledgeGraphBuilder:
    """Classe para construir um grafo de conhecimento do universo Star Wars"""
    
    def __init__(self):
        self.setup_environment()
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.graph = self.setup_neo4j_connection()
        self.transformer = self.setup_graph_transformer()
        
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
    
    def setup_graph_transformer(self) -> LLMGraphTransformer:
        """Configura o transformer com domínio Star Wars expandido"""
        return LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Personagem", "Filme", "Planeta", "Nave", "Facção", 
                "Espécie", "Droide", "Arma", "Tecnologia", "Evento",
                "Localização", "Organização"
            ],
            allowed_relationships=[
                "APARECE_EM", "PILOTA", "PERTENCE_A", "TEM_CENA_EM", 
                "É_CONTROLADO_POR", "NASCEU_EM", "VIVE_EM", "LUTA_CONTRA",
                "ALIADO_DE", "FILHO_DE", "MESTRE_DE", "POSSUI", "LOCALIZADO_EM",
                "ACONTECE_EM", "PARTICIPA_DE", "CRIADO_POR", "USADO_POR"
            ],
            node_properties=[
                "nome", "título", "descrição", "ano", "diretor", "população",
                "clima", "tipo", "fabricante", "modelo", "classe", "lado_da_força",
                "planeta_natal", "espécie", "afiliação", "status"
            ],
            relationship_properties=[
                "tipo", "duração", "importância", "período", "resultado"
            ]
        )
    
    def clear_existing_data(self):
        """Limpa dados existentes no grafo (opcional)"""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            logger.info("🗑️ Dados existentes removidos")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao limpar dados: {e}")
    
    def create_constraints_and_indexes(self):
        """Cria restrições e índices para otimizar performance"""
        constraints = [
            "CREATE CONSTRAINT personagem_nome IF NOT EXISTS FOR (p:Personagem) REQUIRE p.nome IS UNIQUE",
            "CREATE CONSTRAINT filme_titulo IF NOT EXISTS FOR (f:Filme) REQUIRE f.título IS UNIQUE",
            "CREATE CONSTRAINT planeta_nome IF NOT EXISTS FOR (p:Planeta) REQUIRE p.nome IS UNIQUE",
            "CREATE CONSTRAINT nave_nome IF NOT EXISTS FOR (n:Nave) REQUIRE n.nome IS UNIQUE",
            "CREATE CONSTRAINT faccao_nome IF NOT EXISTS FOR (f:Facção) REQUIRE f.nome IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX personagem_nome_idx IF NOT EXISTS FOR (p:Personagem) ON (p.nome)",
            "CREATE INDEX filme_ano_idx IF NOT EXISTS FOR (f:Filme) ON (f.ano)",
            "CREATE INDEX planeta_clima_idx IF NOT EXISTS FOR (p:Planeta) ON (p.clima)"
        ]
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
                logger.info(f"✅ Constraint criada: {constraint.split()[-1]}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao criar constraint: {e}")
        
        for index in indexes:
            try:
                self.graph.query(index)
                logger.info(f"✅ Índice criado: {index.split()[-1]}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao criar índice: {e}")
    
    def load_and_process_documents(self, pdf_path: str) -> List[Document]:
        """Carrega e processa documentos PDF"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
        logger.info(f"📄 Carregando PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        logger.info(f"📄 {len(docs)} páginas carregadas")
        
        # Configuração otimizada para Star Wars
        splitter = CharacterTextSplitter(
            chunk_size=1000,  # Chunks menores para melhor precisão
            chunk_overlap=150,
            separator="\n\n"  # Separação por parágrafos
        )
        chunks = splitter.split_documents(docs)
        
        logger.info(f"📝 {len(chunks)} chunks criados")
        return chunks
    
    def process_chunks_to_graph(self, chunks: List[Document]) -> int:
        """Processa chunks e extrai entidades/relacionamentos"""
        total_entities = 0
        total_relationships = 0
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"🔄 Processando chunk {i+1}/{len(chunks)}")
                
                # Extrai grafo do chunk
                graph_docs = self.transformer.convert_to_graph_documents([chunk])
                
                # Conta entidades e relacionamentos
                for graph_doc in graph_docs:
                    total_entities += len(graph_doc.nodes)
                    total_relationships += len(graph_doc.relationships)
                
                # Adiciona ao Neo4j
                self.graph.add_graph_documents(graph_docs)
                
                # Log do progresso
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Progresso: {i+1}/{len(chunks)} chunks processados")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar chunk {i+1}: {e}")
                continue
        
        logger.info(f"📊 Total extraído: {total_entities} entidades, {total_relationships} relacionamentos")
        return total_entities + total_relationships
    
    def add_manual_knowledge(self):
        """Adiciona conhecimento estruturado manualmente para enriquecer o grafo"""
        manual_data = [
            # Filmes da trilogia original
            """
            CREATE (ep4:Filme {título: 'Episódio IV: Uma Nova Esperança', ano: 1977, diretor: 'George Lucas'})
            CREATE (ep5:Filme {título: 'Episódio V: O Império Contra-Ataca', ano: 1980, diretor: 'Irvin Kershner'})
            CREATE (ep6:Filme {título: 'Episódio VI: O Retorno do Jedi', ano: 1983, diretor: 'Richard Marquand'})
            """,
            
            # Personagens principais
            """
            MERGE (luke:Personagem {nome: 'Luke Skywalker', espécie: 'Humano', planeta_natal: 'Tatooine', lado_da_força: 'Jedi'})
            MERGE (leia:Personagem {nome: 'Leia Organa', espécie: 'Humano', planeta_natal: 'Alderaan', afiliação: 'Rebel Alliance'})
            MERGE (han:Personagem {nome: 'Han Solo', espécie: 'Humano', planeta_natal: 'Corellia', afiliação: 'Rebel Alliance'})
            MERGE (vader:Personagem {nome: 'Darth Vader', espécie: 'Humano', planeta_natal: 'Tatooine', lado_da_força: 'Sith'})
            """,
            
            # Planetas
            """
            MERGE (tatooine:Planeta {nome: 'Tatooine', clima: 'Árido', tipo: 'Deserto', população: 'Baixa'})
            MERGE (alderaan:Planeta {nome: 'Alderaan', clima: 'Temperado', tipo: 'Montanhoso', status: 'Destruído'})
            MERGE (hoth:Planeta {nome: 'Hoth', clima: 'Gelado', tipo: 'Tundra', população: 'Baixa'})
            """,
            
            # Relacionamentos familiares
            """
            MATCH (luke:Personagem {nome: 'Luke Skywalker'})
            MATCH (leia:Personagem {nome: 'Leia Organa'})
            MATCH (vader:Personagem {nome: 'Darth Vader'})
            CREATE (vader)-[:FILHO_DE]->(luke)
            CREATE (vader)-[:FILHO_DE]->(leia)
            CREATE (luke)-[:IRMÃO_DE]->(leia)
            """
        ]
        
        for query in manual_data:
            try:
                self.graph.query(query)
                logger.info("✅ Conhecimento manual adicionado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao adicionar conhecimento manual: {e}")
    
    def validate_graph(self) -> Dict[str, Any]:
        """Valida e fornece estatísticas do grafo criado"""
        stats = {}
        
        # Conta nós por tipo
        node_counts = self.graph.query("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
        """)
        stats['node_counts'] = {row['label']: row['count'] for row in node_counts}
        
        # Conta relacionamentos por tipo
        rel_counts = self.graph.query("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship, count(r) as count
            ORDER BY count DESC
        """)
        stats['relationship_counts'] = {row['relationship']: row['count'] for row in rel_counts}
        
        # Total de nós e relacionamentos
        total_stats = self.graph.query("""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as total_nodes, count(r) as total_relationships
        """)[0]
        stats['totals'] = total_stats
        
        return stats
    
    def run_pipeline(self, pdf_path: str, clear_existing: bool = True):
        """Executa pipeline completo de construção do grafo"""
        try:
            logger.info("🚀 Iniciando pipeline de construção do grafo de conhecimento")
            
            # Limpa dados existentes se solicitado
            if clear_existing:
                self.clear_existing_data()
            
            # Cria constraints e índices
            self.create_constraints_and_indexes()
            
            # Processa documentos
            chunks = self.load_and_process_documents(pdf_path)
            
            # Extrai e insere dados no grafo
            total_extracted = self.process_chunks_to_graph(chunks)
            
            # Adiciona conhecimento manual
            self.add_manual_knowledge()
            
            # Valida e mostra estatísticas
            stats = self.validate_graph()
            
            logger.info("✅ Pipeline concluída com sucesso!")
            logger.info(f"📊 Estatísticas finais:")
            logger.info(f"   - Total de nós: {stats['totals']['total_nodes']}")
            logger.info(f"   - Total de relacionamentos: {stats['totals']['total_relationships']}")
            logger.info(f"   - Nós por tipo: {stats['node_counts']}")
            logger.info(f"   - Relacionamentos por tipo: {stats['relationship_counts']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erro na pipeline: {e}")
            raise

def main():
    """Função principal"""
    builder = StarWarsKnowledgeGraphBuilder()
    
    # Caminho para o PDF
    pdf_path = "docs/pdf.pdf"
    
    # Executa pipeline
    stats = builder.run_pipeline(pdf_path)
    
    print("\n" + "="*50)
    print("🎬 GRAFO DE CONHECIMENTO STAR WARS CRIADO!")
    print("="*50)
    print(f"📊 {stats['totals']['total_nodes']} nós criados")
    print(f"🔗 {stats['totals']['total_relationships']} relacionamentos criados")
    print("\n✨ Pronto para consultas com linguagem natural!")

if __name__ == "__main__":
    main()