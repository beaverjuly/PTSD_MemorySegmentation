from flask import (Blueprint, redirect, render_template, request, session, url_for, current_app)
from .io import write_data, write_metadata

## Initialize blueprint.
bp = Blueprint('experiment', __name__)

def dev_bootstrap_session_from_query():
    """
    DEV ONLY: allow direct entry to /experiment* without going through "/".

    Priority:
    1. Use explicit workerId / assignmentId / hitId if provided.
    2. Fall back to Prolific-style params:
       - PROLIFIC_PID -> workerId
       - SESSION_ID   -> assignmentId
       - STUDY_ID     -> hitId
    3. If dev=1, allow local dummy defaults.

    Frontend developer-mode params (dev, stage, ntrials, block) are handled
    entirely in the HTML/JS via URLSearchParams and need no Python logic here.
    """


    # Do not overwrite an existing session
    if "workerId" in session:
        return

    is_dev = request.args.get("dev") == "1"

    worker_id = (
        request.args.get("workerId")
        or request.args.get("workerID")
        or request.args.get("PROLIFIC_PID")
    )
    assignment_id = (
        request.args.get("assignmentId")
        or request.args.get("SESSION_ID")
        or ""
    )
    hit_id = (
        request.args.get("hitId")
        or request.args.get("STUDY_ID")
        or ""
    )

    # In explicit dev mode, allow safe dummy defaults
    if is_dev and not worker_id:
        worker_id = "dev123"
    if is_dev and not assignment_id:
        assignment_id = "devA"
    if is_dev and not hit_id:
        hit_id = "devH"

    # Still missing a worker id -> let normal error handling occur
    if not worker_id:
        return

    session["workerId"] = worker_id
    session["assignmentId"] = assignment_id
    session["hitId"] = hit_id

    # These must exist because templates and redirect endpoints expect them
    session.setdefault("code_success", current_app.config.get("CODE_SUCCESS", ""))
    session.setdefault("code_reject", current_app.config.get("CODE_REJECT", ""))
    session.setdefault("data", current_app.config.get("DATA_DIR", ""))
    session.setdefault("metadata", current_app.config.get("META_DIR", ""))
    session.setdefault("reject", current_app.config.get("REJECT_DIR", ""))

@bp.route('/experiment')
def experiment():
    dev_bootstrap_session_from_query()
    """Present jsPsych experiment to participant."""

    ## Error-catching: screen for missing session.
    if not 'workerId' in session:

        ## Redirect participant to error (missing workerId).
        return redirect(url_for('error.error', errornum=1000))

    ## Case 1: previously completed experiment.
    elif 'complete' in session:

        ## Update metadata.
        session['WARNING'] = "Revisited experiment page."
        write_metadata(session, ['WARNING'], 'a')

        ## Redirect participant to complete page.
        return redirect(url_for('complete.complete'))

    ## Case 2: repeat visit.
    elif 'experiment' in session and not current_app.config.get("DEBUG_ALLOW_REPEAT", True):
        ## Update participant metadata.
        session['ERROR'] = "1004: Revisited experiment."
        session['complete'] = 'error'
        write_metadata(session, ['ERROR','complete'], 'a')

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1004))

    ## Case 3: first visit.
    else:

        ## Update participant metadata.
        session['experiment'] = True
        write_metadata(session, ['experiment'], 'a')

        ## Present experiment.
        return (
            render_template(
                'experiment.html',
                workerId=session['workerId'],
                assignmentId=session['assignmentId'],
                hitId=session['hitId'],
                code_success=session['code_success'],
                code_reject=session['code_reject'],
            )
        )


# -----------------------------------------------------------------------------
# Additional route: /experiment_noinstr
# -----------------------------------------------------------------------------

@bp.route('/experiment_noinstr')
def experiment_noinstr():
    dev_bootstrap_session_from_query()
    """Present jsPsych experiment without instructions to participant."""

    # Error-catching: screen for missing session.
    if not 'workerId' in session:
        return redirect(url_for('error.error', errornum=1000))

    # Case 1: previously completed experiment.
    if 'complete' in session:
        session['WARNING'] = "Revisited experiment page."
        write_metadata(session, ['WARNING'], 'a')
        return redirect(url_for('complete.complete'))

    # Case 2: repeat visit.
    if 'experiment' in session and not current_app.config.get("DEBUG_ALLOW_REPEAT", True):
        session['ERROR'] = "1004: Revisited experiment."
        session['complete'] = 'error'
        write_metadata(session, ['ERROR', 'complete'], 'a')
        return redirect(url_for('error.error', errornum=1004))

    # Case 3: first visit.
    session['experiment'] = True
    write_metadata(session, ['experiment'], 'a')

    return render_template(
        'experiment_noinstr.html',
        workerId=session['workerId'],
        assignmentId=session['assignmentId'],
        hitId=session['hitId'],
        code_success=session['code_success'],
        code_reject=session['code_reject'],
    )

@bp.route('/experiment', methods=['POST'])
def pass_message():
    """Write jsPsych message to metadata."""

    if request.is_json:

        ## Retrieve jsPsych data.
        msg = request.get_json()

        ## Update participant metadata.
        session['MESSAGE'] = msg
        write_metadata(session, ['MESSAGE'], 'a')

    return ('', 200)

@bp.route('/redirect_success', methods = ['POST'])
def redirect_success():
    """Save complete jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='pass')

    ## Flag experiment as complete.
    session['complete'] = 'success'
    write_metadata(session, ['complete','code_success'], 'a')

    return ('', 200)

@bp.route('/redirect_reject', methods = ['POST'])
def redirect_reject():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'reject'
    write_metadata(session, ['complete','code_reject'], 'a')

    return ('', 200)

@bp.route('/redirect_error', methods = ['POST'])
def redirect_error():
    """Save rejected jsPsych dataset to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON, method='reject')

    ## Flag experiment as complete.
    session['complete'] = 'error'
    write_metadata(session, ['complete'], 'a')

    return ('', 200)
