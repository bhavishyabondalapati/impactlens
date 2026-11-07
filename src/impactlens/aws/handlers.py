"""AWS Lambda entry points. One container image; each function sets its own CMD to one of
these (infra/lambda.tf). They only adapt Lambda's world (environment variables, the bucket,
SSM, /tmp) to the plain step functions in aws/steps.py.

Environment (set by Terraform):
  BUCKET              the private bucket
  STATE_MACHINE_ARN   (start only) the Step Functions workflow to run
  API_KEY_PARAMETER   (report only) SSM SecureString holding the Anthropic key
  MODEL, BUDGET_USD   (report, finish) Claude model and the per-run cost cap
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.parse
from pathlib import Path

from impactlens.aws import steps
from impactlens.aws.store import S3Store


def _store() -> S3Store:
    return S3Store(os.environ["BUCKET"])


def _work() -> Path:
    """A fresh scratch folder. Warm Lambdas keep /tmp between calls, so clear old runs."""
    root = Path(tempfile.gettempdir()) / "impactlens"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def _load_api_key() -> None:
    """Anthropic key from SSM Parameter Store into the environment (once per container).
    The key is set by the owner with `aws ssm put-parameter`; it is never in Terraform."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    import boto3

    name = os.environ["API_KEY_PARAMETER"]
    value = boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
    if not value.startswith("sk-ant-"):
        raise RuntimeError(f"SSM parameter {name} holds no Anthropic key yet: see docs/AWS.md")
    os.environ["ANTHROPIC_API_KEY"] = value


def fetch(event, context):
    return steps.fetch(_store(), event, _work())


def detect(event, context):
    return steps.detect(_store(), event, _work())


def geo_analyze(event, context):
    return steps.geo_analyze(_store(), event, _work())


def retrieve(event, context):
    return steps.retrieve(_store(), event, _work())


def report(event, context):
    _load_api_key()
    return steps.report(_store(), event, _work())


def validate(event, context):
    return steps.validate(_store(), event, _work())


def finish(event, context):
    return steps.finish(_store(), event, _work())


def start(event, context, sfn=None):
    """SQS -> Step Functions. Each SQS message is an S3 "object created" notification for
    requests/<run_id>.json; the run_id becomes the execution name, so the same request file
    cannot start two runs. A failure raises, SQS retries it, and after 3 tries the message
    goes to the dead-letter queue."""
    if sfn is None:
        import boto3

        sfn = boto3.client("stepfunctions")
    started = []
    for message in event.get("Records", []):
        body = json.loads(message["body"])
        for record in body.get("Records", []):  # S3 sends one s3:TestEvent with no Records
            key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
            run_id = Path(key).stem
            if not (key.startswith("requests/") and key.endswith(".json")):
                continue
            if not steps.RUN_ID.match(run_id):
                print(f"skipping {key}: run id must be letters, digits, '-' or '_' (max 64)")
                continue
            sfn.start_execution(stateMachineArn=os.environ["STATE_MACHINE_ARN"], name=run_id,
                                input=json.dumps({"run_id": run_id}))  # fmt: skip
            started.append(run_id)
    print(f"started {started}")
    return {"started": started}
