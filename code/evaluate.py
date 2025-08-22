"""
RAG 시스템 품질 평가 도구 (evaluate.py)

WebUI와 동일한 RAG 방식 및 메모리 방식을 사용하여
RAGAS 프레임워크로 답변 품질을 평가하는 도구입니다.

주요 기능:
- WebUI와 동일한 시스템 초기화 (VectorStoreManager, LLMManager, RetrieverManager, ChatHistoryManager)
- 사전 정의된 질문-정답 데이터셋 사용 (data/eval/question_dataset.json)
- RAGAS 메트릭을 통한 품질 평가 (faithfulness, answer_relevancy, context_recall, answer_correctness)
- 평가 결과 저장 및 리포트 생성 (data/eval/evaluation_results/)
- Upstage API 직접 사용으로 RAGAS 호환성 확보

사용법:
    uv run python code/evaluate.py

"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 현재 스크립트의 디렉토리를 sys.path에 추가
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(script_dir / '.env')

# LangChain 및 Upstage imports
from langchain_upstage import UpstageEmbeddings

# RAGAS imports (Upstage API 사용)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_recall,
    answer_correctness
)
from datasets import Dataset

# 모듈 imports
from modules import (
    VectorStoreManager, LLMManager, RetrieverManager, 
    ChatHistoryManager, LoggerManager, RAGSystemInitializer, RAGQueryProcessor
)


# 전역 설정
project_root = script_dir.parent
dataset_path = project_root / "data" / "eval" / "question_dataset.json"
results_dir = project_root / "data" / "eval" / "evaluation_results"

def setup_upstage_for_ragas():
    """RAGAS에서 Upstage API를 사용하도록 환경변수 설정"""
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if upstage_api_key:
        os.environ["OPENAI_API_KEY"] = upstage_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.upstage.ai/v1"
        os.environ["OPENAI_MODEL_NAME"] = "solar-pro2"




class RAGEvaluator:
    """RAG 시스템 품질 평가를 위한 클래스"""
    
    def __init__(self):
        self.logger = LoggerManager("RAGEvaluator")
        self.results_dir = results_dir
        
        # 결과 디렉토리 생성
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_dataset(self) -> Dict[str, Any]:
        """평가 데이터셋 로드"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def initialize_system(self):
        """RAG 시스템 초기화"""
        result = RAGSystemInitializer.initialize_system(
            current_file_path=script_dir,
            include_sql=False,
            logger_name="EvaluationRAG",
            enable_db_memory=False  # 평가용은 메모리만 사용
        )
        
        if result is None:
            return False
        
        self.vector_manager, self.llm_manager, self.retriever_manager, self.query_processor = result
        return True
    
    def process_questions(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """질문들을 처리하고 답변 생성"""
        results = []
        questions = dataset["questions"]
        
        for i, question_data in enumerate(questions, 1):
            start_time = time.time()
            
            try:
                question = question_data["question"]
                
                # 자동 메모리 기능이 포함된 간단한 질의 처리 사용
                result = self.query_processor.query(
                    question=question,
                    return_sources=True
                )
                
                if not result["success"]:
                    raise Exception(result["error"])
                
                response = result["response"]
                documents = result.get("documents", [])
                source_documents = result.get("sources", [])
                contexts = [doc.page_content for doc in documents]
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # 결과 저장
                results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "generated_answer": response,
                    "ground_truth": question_data["ground_truth"],
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "retrieved_contexts": contexts,
                    "source_documents": source_documents,
                    "expected_sources": question_data.get("expected_sources", []),
                    "processing_time_ms": round(processing_time, 2),
                    "depends_on": question_data.get("depends_on"),
                    "keywords": question_data.get("keywords", [])
                })
                
            except Exception as e:
                # 실패한 경우에도 빈 결과 추가
                results.append({
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "generated_answer": f"오류 발생: {str(e)}",
                    "ground_truth": question_data["ground_truth"],
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "retrieved_contexts": [],
                    "source_documents": [],
                    "expected_sources": question_data.get("expected_sources", []),
                    "processing_time_ms": 0,
                    "error": str(e)
                })
        
        return results
    
    def run_ragas_evaluation(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """RAGAS 메트릭을 사용한 평가 실행 (Upstage API 사용)"""
        self.logger.log_function_start("run_ragas_evaluation")
        
        try:
            # RAGAS 평가를 위한 데이터셋 구성
            dataset_dict = {
                "question": [r["question"] for r in results],
                "answer": [r["generated_answer"] for r in results],
                "contexts": [r["retrieved_contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results]
            }
            
            # Dataset 객체 생성
            dataset = Dataset.from_dict(dataset_dict)
            
            self.logger.log_step("RAGAS 평가 실행", "메트릭: faithfulness, answer_relevancy, context_recall, answer_correctness (Upstage API 사용)")
            
            # baseline.py 방식으로 Upstage 모델 직접 사용
            from langchain_upstage import ChatUpstage, UpstageEmbeddings
            
            upstage_llm = ChatUpstage(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                model="solar-pro2",
                reasoning_effort="high"
            )
            
            upstage_embeddings = UpstageEmbeddings(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                model="embedding-query"
            )
            
            # RAGAS 평가 실행 (baseline.py와 동일한 Upstage 모델 사용)
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, answer_correctness],
                llm=upstage_llm,
                embeddings=upstage_embeddings
            )
            
            # 결과를 딕셔너리로 변환 (baseline.py 방식으로 수정)
            scores = {}
            scores_dict = evaluation_result._scores_dict
            
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
                if metric_name in scores_dict and len(scores_dict[metric_name]) > 0:
                    # 리스트의 평균값 계산
                    score_values = scores_dict[metric_name]
                    if isinstance(score_values, list):
                        # numpy scalar이나 float 처리
                        avg_score = sum(float(v.item()) if hasattr(v, 'item') else float(v) for v in score_values) / len(score_values)
                        scores[metric_name] = avg_score
                    else:
                        scores[metric_name] = float(score_values.item()) if hasattr(score_values, 'item') else float(score_values)
                else:
                    scores[metric_name] = 0.0
            
            # RAGAS 종합 점수 계산 (평균)
            scores["ragas_score"] = sum(scores.values()) / len(scores)
            
            self.logger.log_function_end("run_ragas_evaluation", f"평가 완료: {scores['ragas_score']:.3f}")
            return scores
            
        except Exception as e:
            self.logger.log_error("run_ragas_evaluation", e)
            # 평가 실패 시 기본값 반환
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "ragas_score": 0.0,
                "error": str(e)
            }
    
    def calculate_category_scores(self, results: List[Dict[str, Any]], 
                                overall_scores: Dict[str, float]) -> Dict[str, Any]:
        """카테고리별 점수 계산"""
        self.logger.log_function_start("calculate_category_scores")
        
        categories = {}
        
        for result in results:
            category = result["category"]
            if category not in categories:
                categories[category] = {
                    "questions": [],
                    "question_count": 0
                }
            
            categories[category]["questions"].append(result)
            categories[category]["question_count"] += 1
        
        # 각 카테고리별 점수 (전체 점수 기반 추정)
        category_scores = {}
        for category, data in categories.items():
            category_scores[category] = {
                "question_count": data["question_count"],
                "avg_processing_time_ms": sum(q.get("processing_time_ms", 0) 
                                            for q in data["questions"]) / data["question_count"],
                "success_rate": len([q for q in data["questions"] 
                                   if not q.get("error")]) / data["question_count"]
            }
        
        self.logger.log_function_end("calculate_category_scores", 
                                   f"{len(categories)}개 카테고리 분석")
        return category_scores
    
    def save_evaluation_results(self, dataset: Dict[str, Any], results: List[Dict[str, Any]], 
                              overall_scores: Dict[str, float], category_scores: Dict[str, Any]) -> str:
        """평가 결과 저장"""
        self.logger.log_function_start("save_evaluation_results")
        
        try:
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}.json"
            filepath = self.results_dir / filename
            
            # 결과 데이터 구성
            evaluation_report = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_version": dataset["metadata"]["version"],
                    "model_config": {
                        "llm_model": "solar-pro2",
                        "embedding_model": "embedding-query",
                        "chunk_size": 1000,
                        "chunk_overlap": 50,
                        "retrieval_k": 5
                    },
                    "evaluation_framework": "RAGAS 0.3.2"
                },
                "overall_scores": overall_scores,
                "category_scores": category_scores,
                "detailed_results": results,
                "summary": {
                    "total_questions": len(results),
                    "total_processing_time_ms": sum(r.get("processing_time_ms", 0) for r in results),
                    "avg_processing_time_ms": sum(r.get("processing_time_ms", 0) for r in results) / len(results),
                    "memory_test_count": len([r for r in results if r["category"] == "memory"]),
                    "error_count": len([r for r in results if r.get("error")])
                }
            }
            
            # 결과 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
            
            # latest.json 심볼릭 링크 업데이트
            latest_path = self.results_dir / "latest.json"
            if latest_path.exists():
                latest_path.unlink()
            
            # 상대 경로로 심볼릭 링크 생성
            latest_path.symlink_to(filename)
            
            self.logger.log_function_end("save_evaluation_results", f"결과 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.log_error("save_evaluation_results", e)
            raise
    
    def print_evaluation_summary(self, overall_scores: Dict[str, float], 
                               category_scores: Dict[str, Any], 
                               results: List[Dict[str, Any]]):
        """평가 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎯 RAG 시스템 품질 평가 결과")
        print("="*60)
        
        # 전체 점수
        print("\n📊 전체 RAGAS 점수:")
        print(f"  • Faithfulness (사실 정확성):     {overall_scores.get('faithfulness', 0):.3f}")
        print(f"  • Answer Relevancy (답변 관련성):  {overall_scores.get('answer_relevancy', 0):.3f}")
        print(f"  • Context Recall (컨텍스트 회상률): {overall_scores.get('context_recall', 0):.3f}")
        print(f"  • Answer Correctness (답변 정확성): {overall_scores.get('answer_correctness', 0):.3f}")
        print(f"  • 📈 RAGAS 종합 점수:            {overall_scores.get('ragas_score', 0):.3f}")
        
        # 카테고리별 점수
        print("\n📂 카테고리별 분석:")
        for category, scores in category_scores.items():
            print(f"  • {category.upper()}: {scores['question_count']}개 질문")
            print(f"    - 성공률: {scores['success_rate']:.1%}")
            print(f"    - 평균 처리시간: {scores['avg_processing_time_ms']:.0f}ms")
        
        # 메모리 테스트 결과
        memory_questions = [r for r in results if r["category"] == "memory"]
        if memory_questions:
            memory_success = len([r for r in memory_questions if not r.get("error")])
            print(f"\n🧠 메모리 기능 테스트: {memory_success}/{len(memory_questions)} 성공")
        
        # 처리 통계
        total_time = sum(r.get("processing_time_ms", 0) for r in results)
        avg_time = total_time / len(results) if results else 0
        print(f"\n⏱️  처리 시간 통계:")
        print(f"  • 총 처리시간: {total_time:.0f}ms")
        print(f"  • 평균 처리시간: {avg_time:.0f}ms")
        
        print("\n" + "="*60)
    
    def run_evaluation(self):
        """전체 평가 프로세스 실행"""
        self.logger.log_success("=== RAG 품질 평가 시작 ===")
        
        try:
            # 1. 데이터셋 로드
            dataset = self.load_evaluation_dataset()
            
            # 2. 시스템 초기화
            if not self.initialize_system():
                self.logger.log_error_with_icon("시스템 초기화 실패")
                return False
            
            # 3. 질문 처리 및 답변 생성
            results = self.process_questions(dataset)
            
            # 4. RAGAS 평가 실행
            overall_scores = self.run_ragas_evaluation(results)
            
            # 5. 카테고리별 분석
            category_scores = self.calculate_category_scores(results, overall_scores)
            
            # 6. 결과 저장
            result_file = self.save_evaluation_results(dataset, results, overall_scores, category_scores)
            
            # 7. 결과 출력
            self.print_evaluation_summary(overall_scores, category_scores, results)
            
            print(f"\n💾 상세 결과가 저장되었습니다: {result_file}")
            print(f"📋 최신 결과 확인: {self.results_dir}/latest.json")
            
            self.logger.log_success("=== RAG 품질 평가 완료 ===")
            return True
            
        except Exception as e:
            self.logger.log_error("run_evaluation", e)
            print(f"\n❌ 평가 중 오류 발생: {str(e)}")
            return False


def main():
    """메인 함수"""
    print("🚀 RAG 시스템 품질 평가 CLI")
    print("WebUI와 동일한 RAG 방식으로 평가를 수행합니다.\n")
    
    try:
        # RAGAS 설정
        setup_upstage_for_ragas()
        
        # 평가기 생성 및 실행
        evaluator = RAGEvaluator()
        success = evaluator.run_evaluation()
        
        if success:
            print("\n✅ 평가가 성공적으로 완료되었습니다!")
            return 0
        else:
            print("\n❌ 평가 중 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())