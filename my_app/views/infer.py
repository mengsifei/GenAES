from flask import render_template, session, request, redirect, url_for, flash
from . import infer_bp
import secrets
from ..database import History, db
from flask_login import current_user
from inference.infer_single import * 
from utils.util_func import *
@infer_bp.route('/infer', methods=['GET'])
def infer():
    token = request.args.get('token')
    if token is None:
        flash("Please go to the home page to check your essay.")
        return redirect(url_for('infer.index'))
    if token != session.get('inference_token'):
        flash("Invalid or expired access token. Please submit the form again.")
        return redirect(url_for('infer.index'))
    scores = session.get('scores', [])
    topic = session.get('topic', "")
    essay = session.get('essay', "")
    feedback = session.get('feedback', {}) or {}
    feedback = dict(feedback)
    tr_feedback = feedback.get('TR_feedback', 'No feedback available. Your essay does not make sense, try to refine it.')
    cc_feedback = feedback.get('CC_feedback', 'No feedback available. Your essay does not make sense, try to refine it.')
    lr_feedback = feedback.get('LR_feedback', 'No feedback available. Your essay does not make sense, try to refine it.')
    gra_feedback = feedback.get('GRA_feedback', 'No feedback available. Your essay does not make sense, try to refine it.')
    corrected_essay = feedback.get('Corrected_essay', "No suggestions available. Your essay does not make sense, try to refine it.")
    essay = essay.replace('\n', '<br>')
    corrected_essay = corrected_essay.replace('\\n\\n', '\\n').replace('\\n', '<br>')
    session.pop('inference_token', None)
    return render_template('infer.html', result=scores, topic=topic, essay=essay, tr=tr_feedback, cc=cc_feedback, lr=lr_feedback, gra=gra_feedback, corrected_essay=corrected_essay)

@infer_bp.route('/', methods=['GET', 'POST'])
def index():
    set_seed(42)
    if request.method == 'POST':
        topic = request.form['topic'].strip()
        essay = request.form['essay'].strip()
        if not topic or not essay:
            flash('Please provide both a topic and an essay text.')
            return render_template('index.html', isempty='True')
        num_words = len(essay.split())
        if num_words < 20:
            session['scores'] = [1, 1, 1, 1]
            session['topic'] = topic
            session['essay'] = essay
            session['feedback'] = {}
        else:
            scores, feedback = generate_and_score_essay(
                topic, essay
            )
            session['scores'] = scores
            session['topic'] = topic
            session['essay'] = essay.replace('\\n', '\n')
            session['feedback'] = feedback or {}  # Ensure feedback is at least an empty dict
        token = secrets.token_urlsafe(16)
        session['inference_token'] = token
        if current_user.is_authenticated:
            scores = [int(score) for score in session['scores']]
            new_history = History(
                user_id=current_user.id,
                topic=topic,
                essay=essay,
                score_tr=scores[0],
                score_cc=scores[1],
                score_lr=scores[2],
                score_gra=scores[3]
            )
            db.session.add(new_history)
            db.session.commit()
        
        return redirect(url_for('infer.infer', token=token))
    return render_template('index.html', isempty='False', username=current_user.nickname if current_user.is_authenticated else '')

@infer_bp.route('/rubric_explanation')
def rubric_explanation():
    return render_template('rubric_explanation.html')

@infer_bp.route('/rubric_explanation/task_response')
def task_response():
    return render_template('task_response.html')

@infer_bp.route('/rubric_explanation/coherence_cohesion')
def coherence_cohesion():
    return render_template('coherence_cohesion.html')

@infer_bp.route('/rubric_explanation/lexical_resource')
def lexical_resource():
    return render_template('lexical_resource.html')

@infer_bp.route('/rubric_explanation/grammatical_range_accuracy')
def grammatical_range_accuracy():
    return render_template('grammatical_range_accuracy.html')
