from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask import render_template
from transformers import LongformerTokenizer, AutoTokenizer, AutoModelForCausalLM, LongformerConfig
from models.model import *
from utils.util_func import *
from safetensors.torch import load_file

db = SQLAlchemy()
login_manager = LoginManager()

MODELS_LOADED = False
LONGFORMER_TOKENIZER = None
LONGFORMER_MODEL = None
QWEN_TOKENIZER = None
QWEN_MODEL = None
MODEL_SESSION = None

def load_models():
    global MODELS_LOADED, LONGFORMER_TOKENIZER, LONGFORMER_MODEL, QWEN_TOKENIZER, QWEN_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODELS_LOADED:
        LONGFORMER_TOKENIZER = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        config = LongformerConfig.from_json_file("checkpoints/Longformer_checkpoint/config.json")
        LONGFORMER_MODEL = CustomLongformerForSequenceClassification(config)
        state_dict = load_file("checkpoints/Longformer_checkpoint/model.safetensors", device=device)
        LONGFORMER_MODEL.load_state_dict(state_dict)
        LONGFORMER_MODEL.eval()
        
        model_name = 'Qwen/Qwen3-1.7B'
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained(model_name, device='auto')
        QWEN_TOKENIZER.pad_token_id = QWEN_TOKENIZER.eos_token_id
        QWEN_MODEL = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).half()
        MODELS_LOADED = True

def create_app():
    set_seed(42)
    load_models()
    app = Flask(__name__)
    app.config.from_pyfile('../configs.py')
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        from .views import auth_bp, dashboard_bp, infer_bp, about_bp, error_bp
        app.register_blueprint(auth_bp)
        app.register_blueprint(dashboard_bp)
        app.register_blueprint(infer_bp)
        app.register_blueprint(about_bp)
        app.register_blueprint(error_bp)
        @app.errorhandler(Exception)
        def handle_all_exceptions(e):
            code = getattr(e, 'code', 500)
            error_message = str(e) if hasattr(e, 'description') else "Something went wrong."
            return render_template('error.html', code=code, error_message=error_message), code
        from .database import User, History
        db.create_all()
    
    return app
