## Deploy Worker

This container image provides a minimal CLI that triggers a GitHub Actions workflow to deploy a BentoML model.

### Environment Variables

- `GITHUB_TOKEN` (default) or custom variable referenced via `--token-env`
- Optional: `GITHUB_API_URL` when targeting GitHub Enterprise

### Example usage

```bash
docker build -t deploy-worker containers/deploy
docker run --rm \
  -e GITHUB_TOKEN=ghp_exampletoken \
  deploy-worker \
  --repository my-org/mlops-repo \
  --workflow deploy.yml \
  --ref main \
  --model-name churn-predictor \
  --model-version 5 \
  --artifact-uri s3://bucket/models/churn_v5.pkl \
  --bentoml-target churn_service \
  --deploy-environment production
```
