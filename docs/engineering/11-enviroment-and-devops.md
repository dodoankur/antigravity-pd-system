# 11. Environment & DevOps

## Environments
- **Local:** Docker Compose for dev services.
- **Staging:** Automated deploys from `develop` branch to AWS ECS.
- **Production:** Manual approval deploys from `main`.

## CI/CD Pipeline (GitHub Actions)
1. **Linting/Types:** `eslint` and `tsc`.
2. **Unit Tests:** `jest` and `pytest`.
3. **Build:** Docker image creation and push to ECR.
4. **Deploy:** Terraform apply (for infra) or ECS update.
