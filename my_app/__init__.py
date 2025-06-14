from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask import render_template

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../config.py')
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
        from .models import User, History
        db.create_all()
    
    return app
