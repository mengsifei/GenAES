from flask import Blueprint

auth_bp = Blueprint('auth', __name__, template_folder='templates')
dashboard_bp = Blueprint('dashboard', __name__, template_folder='templates')
infer_bp = Blueprint('infer', __name__, template_folder='templates')
about_bp = Blueprint('about', __name__, template_folder='templates')
error_bp = Blueprint('error', __name__, template_folder='templates')

from . import auth, dashboard, infer, about, error
