try:
    from app.services.sarvam_service import sarvam_service, SarvamService
    from app.services.isolated_inference_service import isolated_inference_service, IsolatedInferenceService
    from app.services.sentence_inference_service import sentence_inference_service, SentenceInferenceService
    from app.services.hackathon_demo_service import hackathon_demo_service, HackathonDemoService
    from app.services.five_class_bigru_service import five_class_bigru_service, FiveClassBiGRUService
except ModuleNotFoundError:
    from backend.app.services.sarvam_service import sarvam_service, SarvamService
    from backend.app.services.isolated_inference_service import isolated_inference_service, IsolatedInferenceService
    from backend.app.services.sentence_inference_service import sentence_inference_service, SentenceInferenceService
    from backend.app.services.hackathon_demo_service import hackathon_demo_service, HackathonDemoService
    from backend.app.services.five_class_bigru_service import five_class_bigru_service, FiveClassBiGRUService

__all__ = [
    "sarvam_service",
    "SarvamService",
    "isolated_inference_service",
    "IsolatedInferenceService",
    "sentence_inference_service",
    "SentenceInferenceService",
    "hackathon_demo_service",
    "HackathonDemoService",
    "five_class_bigru_service",
    "FiveClassBiGRUService",
]
