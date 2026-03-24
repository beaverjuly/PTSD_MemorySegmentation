from flask import (Blueprint, redirect, render_template, request, session, url_for, current_app)
from .io import write_data, write_metadata

## Initialize blueprint.
bp = Blueprint('experiment', __name__)

def dev_bootstrap_session_from_query():
    """
    DEV ONLY: allow direct entry to /experiment* without going through "/".
    Uses PROLIFIC_PID as workerId and populates expected session keys.
    """
    if not current_app.config.get("DEBUG_ALLOW_REPEAT", False):
        return

    if "workerId" not in session:
        pid = request.args.get("PROLIFIC_PID")
        if not pid:
            return  # still missing -> will hit error 1000 as usual

        session["workerId"] = pid
        session["assignmentId"] = request.args.get("SESSION_ID", "")
        session["hitId"] = request.args.get("STUDY_ID", "")

        # These must exist because your templates and redirect endpoints expect them
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
        #return render_template(
         #   'experiment.html', workerId=session['workerId'], assignmentId=session['assignmentId'], hitId=session['hitId'], code_success=session['code_success'], code_reject=session['code_reject'])


# -----------------------------------------------------------------------------
# Additional route: /experiment_noinstr
#
# Some experiment deployments provide a version without instructions.  This
# handler duplicates the logic of the main ``/experiment`` endpoint but
# renders the ``experiment_noinstr.html`` template instead.  If the user has
# already completed the experiment or previously visited the experiment
# endpoint, they will be redirected or flagged as appropriate.

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
    # Update participant metadata to indicate they are starting the experiment.
    session['experiment'] = True
    write_metadata(session, ['experiment'], 'a')

    # Present the no-instructions experiment template.  Pass through
    # identifiers and completion codes to the page.  The html file must
    # exist under app/templates.
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

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
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

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
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

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
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

    ## DEV NOTE:
    ## This function returns the HTTP response status code: 200
    ## Code 200 signifies the POST request has succeeded.
    ## The corresponding jsPsych function handles the redirect.
    ## For a full list of status codes, see:
    ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    return ('', 200)


# @bp.route('/incomplete_save', methods=['POST'])
# def incomplete_save():
#     """Save incomplete jsPsych dataset to disk."""

#     if request.is_json:

#         ## Retrieve jsPsych data.
#         JSON = request.get_json()

#         ## Save jsPsch data to disk.
#         write_data(session, JSON, method='incomplete')

#     ## Flag partial data saving.
#     session['MESSAGE'] = 'incomplete dataset saved'
#     write_metadata(session, ['MESSAGE'], 'a')

#     ## DEV NOTE:
#     ## This function returns the HTTP response status code: 200
#     ## Code 200 signifies the POST request has succeeded.
#     ## For a full list of status codes, see:
#     ## https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
#     return ('', 200)