from flask import render_template
from . import about_bp

@about_bp.route('/about_us')
def about_us():
    return render_template('about_us.html')
