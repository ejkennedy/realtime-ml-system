terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
  }

  backend "s3" {
    bucket = "fraud-detection-tfstate"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = "1.30"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    # General workloads (Flink, feature store)
    general = {
      instance_types = ["m5.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }

    # Serving workloads (Ray Serve — memory optimised)
    serving = {
      instance_types = ["c5.4xlarge"]
      min_size       = 2
      max_size       = 20
      desired_size   = 4
      labels = {
        workload = "serving"
      }
      taints = [{
        key    = "workload"
        value  = "serving"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  tags = local.common_tags
}

# ── ElastiCache Redis (Feast online store) ────────────────────────────────────
module "redis" {
  source = "./modules/elasticache"

  cluster_id         = "${var.project_name}-${var.environment}"
  node_type          = "cache.r6g.large"
  num_cache_nodes    = 1
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  tags = local.common_tags
}

# ── MSK (Managed Kafka / Redpanda alternative) ────────────────────────────────
module "kafka" {
  source = "./modules/msk"

  cluster_name    = "${var.project_name}-${var.environment}"
  kafka_version   = "3.6.0"
  broker_count    = 3
  instance_type   = "kafka.m5.large"
  ebs_volume_size = 100
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  tags = local.common_tags
}

# ── S3 buckets for Iceberg + MLflow ──────────────────────────────────────────
module "s3" {
  source = "./modules/s3-iceberg"

  project_name = var.project_name
  environment  = var.environment

  tags = local.common_tags
}

# ── VPC ──────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-${var.environment}"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = var.environment != "prod"

  tags = local.common_tags
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-redis"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name   = "${var.project_name}-${var.environment}-redis"
  vpc_id = module.vpc.vpc_id
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}
