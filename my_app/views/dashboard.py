from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from . import dashboard_bp
from ..database import History, db


@dashboard_bp.route('/dashboard', defaults={'page': 1})
@dashboard_bp.route('/dashboard/page/<int:page>')
@login_required
def dashboard(page):
    per_page = 10
    pagination = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    histories = pagination.items

    scores_tr = [history.score_tr for history in histories]
    scores_cc = [history.score_cc for history in histories]
    scores_lr = [history.score_lr for history in histories]
    scores_gra = [history.score_gra for history in histories]
    creation_times = [history.created_at.strftime('%Y-%m-%d %H:%M:%S') for history in histories]

    return render_template(
        'dashboard.html',
        histories=histories,
        pagination=pagination,
        scores_tr=scores_tr,
        scores_cc=scores_cc,
        scores_lr=scores_lr,
        scores_gra=scores_gra,
        creation_times=creation_times
    )
@dashboard_bp.route('/delete_histories', methods=['POST'])
@login_required
def delete_histories():
    history_ids = request.form.getlist('history_ids')
    History.query.filter(History.id.in_(history_ids), History.user_id == current_user.id).delete(synchronize_session=False)
    db.session.commit()
    flash('Selected histories deleted.')
    return redirect(url_for('dashboard.dashboard'))
