from . import error_bp
from flask import render_template

@error_bp.errorhandler(Exception)
def handle_all_exceptions(e):
    code = getattr(e, 'code', 500)
    error_message = str(e) if hasattr(e, 'description') else "Something went wrong."
    return render_template('error.html', code=code, error_message=error_message), code