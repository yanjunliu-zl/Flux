# Seedance 2.5: Global Deployment & Billing System Design

> This document defines the architecture, multi-tenancy system, billing model, and financial system design for deploying Seedance 2.5 as a global SaaS/PaaS product, providing video generation API services for enterprise clients and developers.

---

## 1. Product Positioning & Service Models

### 1.1 Three-Tier Product Line

```
┌─────────────────────────────────────────────────────────────────┐
│                    Seedance Cloud                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 1: Web UI (Creator Tool)                             │ │
│  │  Browser → prompt + reference → generate → download         │ │
│  │  Users: Content creators, marketers, independent artists    │ │
│  │  Pricing: Subscription ($29-299/mo) + overage pay-per-use  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 2: API (Developer Platform)                           │ │
│  │  REST/gRPC API → generate → Webhook callback                │ │
│  │  Users: Startups, AI app developers, video tool integrations│ │
│  │  Pricing: Pure usage-based ($/sec video), tiered discounts  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 3: Enterprise (Private Deployment)                   │ │
│  │  Dedicated cluster → custom models → data isolation → certs │ │
│  │  Users: Major media cos, game studios, government/military  │ │
│  │  Pricing: Annual contract ($500K-5M/yr) w/ SLA + support   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Generation Specs & Pricing Anchors

| Spec | Codename | Target Latency | Resource (GPU·s) | Base Price (per video) |
|------|----------|---------------|-------------------|------------------------|
| **Fast** — 480p, 2s, 16fps, no audio | `flux-fast` | <30s | 8 | $0.08 |
| **Standard** — 720p, 5s, 24fps, with audio | `flux-std` | <60s | 30 | $0.30 |
| **Pro** — 1080p, 10s, 30fps, with audio | `flux-pro` | <120s | 120 | $1.20 |
| **Max** — 4K, 30s, 30fps, with audio | `flux-max` | <180s | 600 | $6.00 |
| **Ultra** — 4K, 60s, 30fps, with audio, 50 ref inputs | `flux-ultra` | <360s | 1500 | $15.00 |

---

## 2. Global Deployment Architecture

### 2.1 Region Planning

```
                           ┌──────────────┐
                           │   Global     │
                           │   Control    │
                           │   Plane      │
                           │   (us-east)  │
                           └──────┬───────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        ▼             ▼           ▼           ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │US East  │  │US West  │  │ Europe  │  │ APAC    │  │ Middle  │
   │us-east  │  │us-west  │  │eu-west  │  │ap-east  │  │East     │
   │Virginia │  │Oregon   │  │Frankfurt│  │Singapore│  │me-east  │
   │8×H100   │  │4×H100   │  │8×H100   │  │8×H100   │  │Dubai    │
   └─────────┘  └─────────┘  └─────────┘  └─────────┘  │4×H100   │
        │             │           │           │        └─────────┘
        └─────────────┴───────────┴───────────┴─────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │S. America│ │ Africa  │  │S. Asia  │
              │sa-east  │  │af-south │  │in-west  │
              │São Paulo│  │Cape Town│  │Mumbai   │
              │CDN only │  │CDN only │  │4×H100   │
              └─────────┘  └─────────┘  └─────────┘
```

| Region | Code | GPU Nodes | Purpose | Compliance |
|--------|------|-----------|---------|------------|
| **US East** | us-east-1 | 8×H100 | Primary training + inference + control plane | SOC2, HIPAA |
| **US West** | us-west-2 | 4×H100 | Inference (West Coast low latency) | SOC2 |
| **Europe** | eu-central-1 | 8×H100 | Inference + data residency | GDPR, EU AI Act |
| **APAC** | ap-southeast-1 | 8×H100 | Inference + Asian customers | PDPA (SG) |
| **Middle East** | me-central-1 | 4×H100 | Inference | UAE Data Law |
| **South Asia** | ap-south-1 | 4×H100 | Inference (India market) | DPDP Act 2023 |
| **South America** | sa-east-1 | CDN only | Edge caching | LGPD (BR) |
| **Africa** | af-south-1 | CDN only | Edge caching | POPIA (ZA) |

### 2.2 Regional Deployment Tiers

```
Region Deployment Tiers:

┌─────────────────────────────────────────────────────────────────┐
│ Tier 1 Region (Full Stack): us-east, eu-central, ap-southeast  │
│   ├── GPU inference cluster (8×H100)                            │
│   ├── Model registry + weight storage                           │
│   ├── User data storage (data residency)                        │
│   ├── Local databases (PostgreSQL + Redis)                      │
│   ├── Billing compute nodes                                     │
│   └── Full monitoring + alerting stack                          │
│                                                                  │
│ Tier 2 Region (Inference Only): us-west, me-central, ap-south  │
│   ├── GPU inference cluster (4×H100)                            │
│   ├── Model cache (sync from Tier 1)                            │
│   ├── User data ephemeral (deleted after processing)            │
│   └── Lightweight monitoring                                    │
│                                                                  │
│ Tier 3 Region (Edge/CDN Only): sa-east, af-south               │
│   ├── CDN edge nodes (Cloudflare / Fastly)                     │
│   ├── Static asset caching (result videos, Web UI)             │
│   └── No GPU — requests forwarded to nearest Tier 1/2          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Cross-Region Traffic Routing

```
User Request
  │
  ▼
┌──────────────────────────────────────────────┐
│            Global Traffic Router              │
│  ┌────────────────────────────────────────┐  │
│  │ DNS Geo-steering (Route53 / Cloudflare) │  │
│  │ 1. User IP → nearest region            │  │
│  │ 2. Region health check → failover       │  │
│  │ 3. Data residency check → GDPR routing  │  │
│  │ 4. Capacity-aware → overflow to neighbor│  │
│  └────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│us-east   │    │eu-central│    │ap-se     │
│Normal    │    │Normal    │    │Overloaded│
│→ local   │    │→ local   │    │→ overflow│
│          │    │(GDPR OK) │    │  to      │
│          │    │          │    │  us-west │
└──────────┘    └──────────┘    └──────────┘
```

**Routing Rule Priority**:
1. **Data residency** (highest): GDPR users → force EU Region
2. **Latency optimal**: IP → GeoDNS → nearest available Region
3. **Capacity overflow**: Primary Region queue depth > N → overflow to backup
4. **Cost optimization**: Prefer Spot GPUs when available, on-demand as fallback

---

## 3. Multi-Tenancy System

### 3.1 Tenant Isolation Model

```
┌──────────────────────────────────────────────────────────────┐
│                    Tenant Isolation                           │
│                                                               │
│  Control Plane (Shared)                                       │
│    ├── User management + Auth (Auth0 / Keycloak)              │
│    ├── Billing + Invoicing (Stripe / Custom)                  │
│    ├── Usage metering (per-request metering)                  │
│    └── API Gateway (Kong / Envoy)                             │
│                                                               │
│  Data Plane (Isolated by Tier)                                │
│    ├── Shared (Tier 1): Shared GPU pool + request-level iso   │
│    ├── Dedicated (Tier 2): Namespace isolation + GPU quota    │
│    └── Private (Tier 3): Full physical isolation + dedicated  │
└──────────────────────────────────────────────────────────────┘
```

| Isolation Dimension | Shared (Tier 1) | Dedicated (Tier 2) | Private (Tier 3) |
|--------------------|-----------------|-------------------|-----------------|
| **Compute** | Shared GPU pool, request-level | K8s Namespace, GPU quota | Dedicated GPU nodes |
| **Storage** | Shared S3 bucket, user prefix | Dedicated bucket, IAM policy | Dedicated storage cluster |
| **Network** | Shared VPC | Isolated VPC, PrivateLink | Isolated VPC + VPN/Direct Connect |
| **Data Encryption** | Server-side encryption (SSE-S3) | Independent KMS key | Customer-managed KMS (BYOK) |
| **Audit Logs** | Shared log stream | Independent stream + export | SIEM integration |
| **Compliance Certs** | SOC2 | SOC2 + ISO27001 | Custom (FedRAMP, HIPAA) |

### 3.2 Tenant Lifecycle Management

```
Provision → Active → Upgrade/Downgrade → Suspended → Terminated
   │           │            │                │            │
   ▼           ▼            ▼                ▼            ▼
Create      Normal      Modify quota/    Past-due/     Data deletion
Tenant ID   service     change plan      violation     Compliance
Assign      Usage       Hot-reload       Retain 30d    erase
Region      metering    (no downtime)    Resume→Active Confirm audit
Quota       Billing     Notify user      Permanent
Allocate    generation  Sync quotas      delete
```

---

## 4. Billing System Design

### 4.1 Pricing Models

#### 4.1.1 Pay-as-you-go

| Billing Dimension | Unit | Price | Notes |
|-------------------|------|-------|-------|
| **Generation Time** | GPU·second | $0.04/GPU·s | Billed by actual GPU occupancy, second-level metering |
| **Video Output** | Per video | See §1.2 table | Bundled price based on spec |
| **Storage** | GB·month | $0.02 | Generated videos + source material storage |
| **Bandwidth** | GB egress | $0.05 | Downloads and CDN distribution |

**GPU·second Pricing Detail** (all compute normalized to H100 equivalent):

| Operation | GPU·s Factor | Description |
|-----------|-------------|-------------|
| T5 Encoding | 1× | Text encoding (standard) |
| Coarse Gen (32fr, 256px) | 8× | Stage B |
| Temporal Ext (32→128fr) | 6× | Stage C |
| Spatial SR (256→1080p) | 15× | Stage D (1080p) |
| Spatial SR (256→4K) | 60× | Stage D (4K) |
| Audio Gen | 4× | Stage E |
| Physics Check (PhaseLock) | 1.5× | Optional, improves physics |
| Post-process (encode+watermark) | 0.5× | Stage F |

**Billing Formula**:
```
cost = Σ (operation_gpu_seconds × $0.04 × region_multiplier) + storage + bandwidth

region_multiplier:
  us-east: 1.00 (baseline)
  us-west: 1.05
  eu-central: 1.12
  ap-southeast: 1.15
  ap-south: 0.90
  me-central: 1.08
```

#### 4.1.2 Subscription Plans

| Plan | Monthly | Included | Overage Rate | Features |
|------|---------|----------|-------------|----------|
| **Starter** | $29 | 50 GPU·min (3,000 GPU·s) | $0.05/GPU·s | Standard spec, 10 concurrent, community support |
| **Creator** | $99 | 200 GPU·min (12,000 GPU·s) | $0.045/GPU·s | Pro spec, 50 concurrent, priority queue, email support |
| **Studio** | $299 | 700 GPU·min (42,000 GPU·s) | $0.04/GPU·s | Max spec, 200 concurrent, dedicated queue, Slack support |
| **Business** | $999 | 2,500 GPU·min (150,000 GPU·s) | $0.035/GPU·s | All specs, 500 concurrent, 99.9% SLA, dedicated support |
| **Enterprise** | Custom | Negotiated | Negotiated | Private deploy, custom models, 99.99% SLA, data residency guarantee |

**Subscription Recommendation Logic**:
```
Monthly usage < 3,000 GPU·s   → Starter ($29)
Monthly usage 3K-12K GPU·s    → Creator ($99, save 20%)
Monthly usage 12K-42K GPU·s   → Studio ($299, save 33%)
Monthly usage 42K-150K GPU·s  → Business ($999, save 40%)
Monthly usage > 150K GPU·s    → Enterprise (custom quote, save 50%+)
```

#### 4.1.3 Prepaid Credits

For one-time projects or infrequent use:
- **$10** → 250 GPU·s (no expiry)
- **$50** → 1,375 GPU·s (10% bonus)
- **$200** → 6,000 GPU·s (20% bonus)

#### 4.1.4 Free Tier

| Limit | Value |
|-------|-------|
| Monthly free GPU·s | 300 (~5 Standard videos) |
| Output specs | Fast + Standard |
| Concurrency | 2 |
| Watermark | Mandatory (cannot remove) |
| Storage | 7-day auto-delete |

### 4.2 Metering Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Metering Pipeline                              │
│                                                               │
│  API Gateway (Envoy/Kong)                                    │
│    │ Per request → Request ID, Tenant ID, Region, Timestamp  │
│    │ Inject Header: X-Seedance-Tenant, X-Seedance-Tier       │
│    ▼                                                         │
│  GPU Orchestrator                                            │
│    │ Record GPU alloc time + release time + GPU type         │
│    │ Record actual ops: encode, denoise, decode, sr, audio... │
│    ▼                                                         │
│  Metrics Agent (per GPU node)                                │
│    │ Real-time: GPU utilization, VRAM, per-op latency        │
│    │ Push → Kafka topic: flux.metering.raw               │
│    ▼                                                         │
│  Stream Processor (Apache Flink)                             │
│    │ Window aggregation (1min tumbling window)               │
│    │ Dedup + anomaly detection (negative, oversized, dupes)  │
│    │ Aggregate by Tenant × Region × Operation                │
│    │ Output → Kafka topic: flux.metering.aggregated      │
│    ▼                                                         │
│  Metering Database (ClickHouse)                              │
│    │ Time-series: per-tenant per-second usage                │
│    │ Materialized views: hourly/daily/monthly pre-aggregation│
│    │ Retention: raw 90 days, aggregated 3 years              │
│    ▼                                                         │
│  Billing Engine (Cron, hourly)                               │
│    │ Read ClickHouse → compute charges → write to billing DB │
│    │ Overage detection → notify → auto throttle (optional)   │
│    │ Monthly settlement → generate invoice → Stripe charge   │
│    ▼                                                         │
│  Billing Database (PostgreSQL)                               │
│    │ Invoices, payments, credit balances                     │
│    │ Audit-immutable log (WAL archive → S3)                 │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Metering Data Schema

```sql
-- ClickHouse: Raw Usage Events Table
CREATE TABLE metering.usage_events (
    event_id        UUID,
    tenant_id       String,
    request_id      String,
    region          LowCardinality(String),
    operation       LowCardinality(String),  -- 't5_encode','coarse_gen','temporal_ext',...
    gpu_type        LowCardinality(String),  -- 'H100','A100','L40S'
    gpu_count       UInt8,
    gpu_seconds     Decimal64(3),            -- actual GPU seconds
    h100_equivalent Decimal64(3),             -- normalized to H100 equivalent
    billable        Decimal64(3),             -- billable amount (after free tier)
    request_status  LowCardinality(String),  -- 'success','failed','cancelled'
    error_code      String,
    timestamp       DateTime64(3),
    received_at     DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (tenant_id, timestamp)
TTL timestamp + INTERVAL 90 DAY;

-- Materialized View: Hourly Aggregation
CREATE MATERIALIZED VIEW metering.usage_hourly
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (tenant_id, region, hour)
AS SELECT
    tenant_id,
    region,
    toStartOfHour(timestamp) as hour,
    count() as request_count,
    sum(gpu_seconds) as total_gpu_seconds,
    sum(h100_equivalent) as total_h100_eq,
    sum(billable) as total_billable,
    countIf(request_status = 'failed') as failed_count
FROM metering.usage_events
GROUP BY tenant_id, region, hour;
```

### 4.4 Billing & Invoicing

```
┌─────────────────────────────────────────────────────────────────┐
│  Seedance Cloud — Monthly Statement                             │
│  Billing Period: July 1-31, 2026                                │
│  Account: Acme Studios (tenant_acme_01)                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Subscription: Studio Plan                         $299.00   │ │
│  │   Included: 700 GPU·min (42,000 GPU·s)                      │ │
│  │   Used:     52,400 GPU·s                                    │ │
│  │   Overage:  10,400 GPU·s × $0.040/GPU·s  = $416.00         │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Storage: 142 GB × $0.02/GB/month               $2.84       │ │
│  │ Bandwidth (CDN): 820 GB × $0.05/GB              $41.00      │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Physics Enhancement (PhaseLock): 210 ops                    │ │
│  │   210 × 1.5 GPU·s × $0.040 = $12.60                        │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ Subtotal                                         $771.44    │ │
│  │ Tax (VAT 20%, EU customer)                       $154.29    │ │
│  │ Total                                            $925.73    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Usage Breakdown:                                                │
│  ┌──────────────┬────────┬──────────┬──────────┬──────────────┐ │
│  │ Operation    │ Count  │ GPU·s    │ H100 eq  │ Cost         │ │
│  ├──────────────┼────────┼──────────┼──────────┼──────────────┤ │
│  │ t5_encode    │ 1,203  │ 1,203    │ 1,203    │ $0.00 (incl) │ │
│  │ coarse_gen   │ 1,203  │ 9,624    │ 9,624    │ $0.00 (incl) │ │
│  │ temporal_ext │ 892    │ 5,352    │ 5,352    │ $0.00 (incl) │ │
│  │ spatial_sr   │ 756    │ 11,340   │ 11,340   │ $0.00 (incl) │ │
│  │ audio_gen    │ 523    │ 2,092    │ 2,092    │ $0.00 (incl) │ │
│  │ physics      │ 210    │ 315      │ 315      │ $12.60       │ │
│  └──────────────┴────────┴──────────┴──────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Payment & Invoicing Integration

```
┌──────────────────────────────────────────────────────┐
│                 Payment Architecture                  │
│                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │ Stripe       │   │ Internal    │   │ Enterprise   │ │
│  │ (Primary)    │   │ Credits     │   │ Invoicing    │ │
│  │              │   │             │   │              │ │
│  │ Credit/Debit │   │ Prepaid     │   │ NET-30/60    │ │
│  │ Apple/Google │   │ credits     │   │ PO + Invoice │ │
│  │ Pay          │   │ Bulk buy    │   │ Wire/ACH     │ │
│  │ PayPal       │   │ Promo codes │   │ Custom terms │ │
│  │ Auto-renew   │   │ Internal    │   │              │ │
│  └──────┬───────┘   │ transfers   │   │              │ │
│         │           └──────┬──────┘   └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            ▼                            │
│                  ┌──────────────────┐                    │
│                  │ Billing Service   │                    │
│                  │                   │                    │
│                  │ Invoice gen (PDF) │                    │
│                  │ Tax calculation   │                    │
│                  │ (Stripe Tax or    │                    │
│                  │  Avalara)         │                    │
│                  │ Statement PDF     │                    │
│                  │ Webhook notify    │                    │
│                  │ Payment retry     │                    │
│                  └──────────────────┘                    │
└──────────────────────────────────────────────────────┘
```

---

## 5. Quotas & Rate Limiting

### 5.1 Multi-Level Quota System

```
┌────────────────────────────────────────────────────────────┐
│                   Quota Enforcement                        │
│                                                            │
│  Level 1 — Tenant Quota (User-Level)                       │
│    ├── Monthly GPU·s cap (plan-dependent)                  │
│    ├── Daily GPU·s cap (abuse prevention)                  │
│    ├── Concurrent request cap (plan-dependent)             │
│    └── Storage capacity cap                                │
│                                                            │
│  Level 2 — Region Quota (Regional)                         │
│    ├── Per-region GPU capacity cap                         │
│    ├── Per-region queue depth cap                          │
│    └── Cross-region overflow quota                         │
│                                                            │
│  Level 3 — Global Quota (Global)                           │
│    ├── Total GPU capacity cap                              │
│    ├── Emergency reserve (5% for burst)                    │
│    └── Free tier user total cap                            │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Rate Limiting Policy

```yaml
# Envoy Rate Limit Configuration
rate_limits:
  - name: "tenant_per_second"
    descriptors:
      - key: "tenant_id"
        rate_limit:
          unit: second
          requests_per_unit: 10       # Starter
          requests_per_unit: 50       # Creator
          requests_per_unit: 200      # Studio
          requests_per_unit: 500      # Business

  - name: "tenant_per_day_gpu"
    descriptors:
      - key: "tenant_id"
        rate_limit:
          unit: day
          requests_per_unit: 3000     # Starter (GPU·s)
          requests_per_unit: 12000    # Creator
          requests_per_unit: 42000    # Studio
          requests_per_unit: 150000   # Business

  - name: "global_concurrent"
    descriptors:
      - key: "global"
        rate_limit:
          unit: second
          requests_per_unit: 2000

  - name: "free_tier_global"
    descriptors:
      - key: "tier"
        value: "free"
        rate_limit:
          unit: minute
          requests_per_unit: 100
```

### 5.3 Overage Handling

```
Usage threshold reached:
  │
  ├── 80% monthly quota:  Send reminder email + Dashboard notification
  ├── 95% monthly quota:  Send alert + recommend plan upgrade
  ├── 100% monthly quota: Auto-switch to overage billing ($0.05/GPU·s)
  │                        Or: soft limit (queue at lower priority)
  ├── 150% overage:       Force rate limit (queued, increased latency)
  ├── 200% overage:       Suspend service, requires manual confirmation or upgrade
  │
  └── Enterprise users:   Overage per contract terms
                          Typically: auto-scale + end-of-month settlement
```

---

## 6. Security & Compliance

### 6.1 Data Security Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Data Security                            │
│                                                               │
│  In Transit (TLS 1.3)                                         │
│    ├── Client ↔ API Gateway: mTLS (API users)                │
│    ├── Service-to-service: mTLS + SPIFFE                     │
│    ├── Cross-Region: WireGuard/IPsec tunnel                  │
│    └── Storage access: TLS + IAM                             │
│                                                               │
│  At Rest                                                      │
│    ├── User data (S3): AES-256-GCM, per-tenant KMS key       │
│    ├── Model weights: AES-256, region-scoped KMS key         │
│    ├── Logs/metrics: AES-256, service-level key              │
│    └── Databases: TDE (Transparent Data Encryption)          │
│                                                               │
│  In Use (Confidential Computing, H100)                        │
│    ├── GPU TEE: NVIDIA Confidential Computing (CC mode)      │
│    ├── Memory encryption: cold-boot attack prevention        │
│    └── Attestation: remote attestation of GPU firmware       │
│                                                               │
│  Key Management (HashiCorp Vault / AWS KMS)                   │
│    ├── Auto-rotation (90 days)                               │
│    ├── BYOK support (Enterprise only)                        │
│    ├── HSM hardware protection (FIPS 140-2 Level 3)          │
│    └── Audit log: every key access recorded                  │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Regional Compliance Matrix

| Regulation | Region | Key Requirements | Implementation |
|-----------|--------|-----------------|----------------|
| **GDPR** | EU/EEA | Data residency, right to erasure, DPA | EU Region dedicated, auto-delete API |
| **EU AI Act** | EU | High-risk AI system transparency | Content watermark, provenance label (C2PA) |
| **SOC2 Type II** | Global (Enterprise) | Security/availability/confidentiality | Annual audit, access control, monitoring |
| **CCPA/CPRA** | California | Consumer privacy rights | Data deletion API, do not sell data |
| **LGPD** | Brazil | GDPR-like | sa-east data residency (CDN only) |
| **PDPA** | Singapore | Data protection | ap-southeast compliance |
| **DPDP Act 2023** | India | Data localization | ap-south data residency |
| **HIPAA** | US Healthcare | Health data protection | Enterprise only, BAA signed |

### 6.3 Content Safety

```
Input Safety:
  ├── Prompt injection detection (LLM-based classifier)
  ├── Reference image NSFW detection
  ├── Prohibited content keyword matching
  └── Rate anomaly detection (DoS prevention)

Output Safety:
  ├── Generated video NSFW detection (video frames + CLIP)
  ├── Generated audio harmful content detection
  ├── Invisible watermark (StegaStamp, C2PA metadata)
  ├── Deepfake detection fingerprint
  ├── Copyright similarity detection (vs training set)
  └── Child safety content detection (PhotoDNA/CSAM hashing)

Violation Handling:
  ├── Real-time blocking: requests violating content policy rejected
  ├── Human review queue: borderline cases submitted for review
  ├── Appeal process: users can appeal false positives
  └── User education: provide compliance usage guide
```

---

## 7. SLA & Service Reliability

### 7.1 SLA Definitions

| Tier | Monthly Uptime | Monthly Credit | P50 Latency | P95 Latency | Support Response |
|------|---------------|----------------|-------------|-------------|-----------------|
| **Starter** | 99.0% | Service credits (overage) | None | None | 48h |
| **Creator** | 99.5% | 10% bill credit | <90s (Standard) | <180s | 24h |
| **Studio** | 99.9% | 20% bill credit | <60s (Standard) | <120s | 8h |
| **Business** | 99.95% | 30% bill credit | <45s (Standard) | <90s | 4h |
| **Enterprise** | 99.99% | Negotiable | Custom | Custom | 1h (P1), 4h (P2) |

**SLA Calculation**:
```
Uptime = (total_minutes - unavailable_minutes) / total_minutes × 100%

Unavailable = API returns 5xx error rate > 5% for ≥ 5 continuous minutes

Exclusions:
  - Planned maintenance (72h advance notice)
  - Rate limiting due to user overage
  - Third-party dependency (Stripe, Cloudflare) outages
  - Force majeure events
```

### 7.2 High Availability Architecture

```
Each Tier 1 Region (us-east, eu-central, ap-southeast):

┌─────────────────────────────────────────────────────────────┐
│                     Region HA Design                        │
│                                                              │
│  Availability Zone A           Availability Zone B          │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │ API Gateway×2    │        │ API Gateway×2    │          │
│  │ Load Balancer    │◄──────►│ Load Balancer    │          │
│  │ GPU Node×4       │        │ GPU Node×4       │          │
│  │ PostgreSQL (R/W) │◄──────►│ PostgreSQL (R/O) │          │
│  │ Redis×3          │◄──┬───►│ Redis×3          │          │
│  │ Kafka×3          │   │    │ Kafka×3          │          │
│  └──────────────────┘   │    └──────────────────┘          │
│                          │                                   │
│              ┌───────────┘                                   │
│              ▼                                               │
│  ┌──────────────────┐                                       │
│  │ S3 (Cross-AZ)    │                                       │
│  │ Object Store     │                                       │
│  │ (11 9's durability)│                                     │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Disaster Recovery

| Scenario | RPO | RTO | Recovery Strategy |
|----------|-----|-----|-------------------|
| **Single GPU node failure** | 0 | <2 min | K8s auto-reschedule |
| **Single AZ failure** | 0 | <10 min | Traffic cut over to other AZ |
| **Single Region failure** | <5 min | <30 min | DNS failover to backup Region |
| **Database corruption** | <5 min | <1 hour | Point-in-time recovery (PITR) |
| **Model weight corruption** | 0 | <15 min | S3 version restore + reload |
| **Full Region data loss** | <1 hour | <4 hours | Cross-region backup restore |

---

## 8. Cost Optimization

### 8.1 GPU Resource Optimization

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Spot/Preemptible GPU** | 40-60% | 70% of inference load on Spot, with checkpoint + fast recovery |
| **GPU time-sharing (MIG/MPS)** | 15-30% | H100 MIG partition into independent instances, small tasks share |
| **Model quantization (FP8/INT8)** | 30-50% | H100 FP8 Tensor Cores, quantize at inference |
| **Batching** | 20-40% | Coalesce small requests into batches, slight latency increase for large throughput gain |
| **Model distillation** | 60-80% | 200B Teacher → 7B Student (Fast tier) |
| **Caching** | 10-30% | Same/similar prompt result caching (Redis + FAISS semantic dedup) |
| **Temporal prediction** | 15-25% | Pre-generate + cache for common prompts |

### 8.2 Storage Optimization

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Lifecycle Policies** | 40-60% | Auto hot→cold migration, Free tier 7-day auto-delete |
| **Video Transcode Compression** | 50-70% | Generate raw ProRes → distribute H.265/AV1 |
| **Dedup Storage** | 10-20% | Content-addressable storage; identical content not duplicated |
| **CDN Caching** | 80-90% (bandwidth) | Hot video served from CDN, avoids origin pull |

### 8.3 Network Optimization

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Compression** | 40-60% | Brotli compression on gRPC request/response |
| **Internal routing** | 100% | Cross-AZ → same-Region internal network, avoid public internet charges |
| **Regional CDN** | 50-80% | Static assets (Web UI) deployed to edge nodes |

---

## 9. Key Metrics & Monitoring

### 9.1 Business Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│                    Business Dashboard                         │
│                                                               │
│  Revenue Metrics                   Usage Metrics             │
│  ┌────────────────────┐           ┌────────────────────┐     │
│  │ MRR: $XXX,XXX      │           │ Active Tenants: XK │     │
│  │ ARR: $X,XXX,XXX    │           │ DAU: XXK           │     │
│  │ Churn Rate: X.X%   │           │ GPU·h/day: XX,XXX  │     │
│  │ LTV: $XXX          │           │ Videos/day: XXXK   │     │
│  │ CAC: $XXX          │           │ Avg GPU·s/video: X │     │
│  └────────────────────┘           └────────────────────┘     │
│                                                               │
│  Revenue by Region                  Revenue by Tier          │
│  ┌────────────┬────────┐           ┌────────────────────┐    │
│  │ us-east    │ 35%    │           │ Enterprise: 45%    │    │
│  │ eu-central │ 28%    │           │ Business:   25%    │    │
│  │ ap-se      │ 22%    │           │ Studio:     18%    │    │
│  │ others     │ 15%    │           │ Creator:     8%    │    │
│  └────────────┴────────┘           │ Starter:     4%    │    │
│                                    └────────────────────┘    │
│                                                               │
│  Real-time Alerts                                             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ⚠  eu-central GPU queue depth > 200 (threshold: 150)     │ │
│  │ ✓  Stripe payment success rate = 99.8%                   │ │
│  │ ⚠  Free tier abuse detected: tenant_xxx (3σ above avg)  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 Technical Dashboard

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **GPU Utilization** | >75% | <60% (resource waste) |
| **Model FLOPs Utilization (MFU)** | >50% (H100) | <35% |
| **Request Success Rate** | >99.9% | <99.5% |
| **P95 End-to-End Latency** | <120s (Standard) | >180s |
| **Billing Accuracy** | >99.99% (vs actual usage) | Deviation >0.1% |
| **Metering Event Loss Rate** | <0.01% | >0.1% |
| **Free Tier Abuse Detection Accuracy** | >95% | <90% |
| **Safety Classifier False Positive Rate** | <2% | >5% |

---

## 10. Operations Team

### 10.1 Operations Staffing

| Role | Headcount | Responsibilities |
|------|----------|-----------------|
| **SRE Lead** | 1 | Overall reliability, incident management |
| **SRE (per region)** | 1 | Regional ops (on-call rotation) |
| **Platform Engineer** | 2 | K8s, CI/CD, IaC |
| **Security Engineer** | 1 | Security monitoring, penetration testing, compliance audit |
| **Data/ML Engineer** | 1 | Model updates, performance optimization |
| **Billing Engineer** | 1 | Billing system maintenance, financial reconciliation |
| **Support Engineer** | 2 | Customer tickets (Follow-the-sun) |
| **Total Ops** | **8-10** | |

### 10.2 On-Call Strategy

```
Follow-the-Sun (24/7 coverage):

  08:00-20:00 UTC+8  →  ap-se SRE (Singapore)
  08:00-20:00 UTC+1  →  eu-central SRE (Frankfurt)
  08:00-20:00 UTC-5  →  us-east SRE (Virginia)

  Weekend: rotation, 1 in 4 weeks
  Holidays: double pay + comp time

  Escalation:
    L1 (SRE)      → 15 min response, 60 min resolve or escalate
    L2 (Lead)     → 30 min response, 2h resolve or escalate
    L3 (VP Eng)   → Major incident (>50% users affected or >$10K/hr loss)
```

---

## 11. Appendix: Pricing Calculator API

Public pricing estimation endpoint:

```json
// POST /v1/pricing/estimate
{
  "spec": "pro",
  "num_videos": 100,
  "options": {
    "physics_enhancement": true,
    "duration_s": 10,
    "include_audio": true,
    "region": "auto"
  }
}

// Response
{
  "estimate": {
    "total_cost": "$120.00",
    "breakdown": {
      "generation": "$118.80",
      "physics_enhancement": "$6.00",
      "storage_estimated_monthly": "$2.40",
      "bandwidth_estimated": "$0.00"
    },
    "gpu_seconds_total": 3000,
    "subscription_applied": "studio",
    "subscription_included_remaining": "39,000 GPU·s",
    "currency": "USD",
    "valid_until": "2026-07-24T00:00:00Z"
  }
}
```

---

## 12. Summary

Deploying Seedance 2.5 globally as a commercial product centers on the **reliability + billing accuracy + compliance** triangle. Key architectural decisions:

1. **3 Full-Stack Regions + CDN global coverage**: 8 regions worldwide, 3 Full Stack for core capabilities, edge CDN for distribution
2. **Second-level metering + ClickHouse time-series storage**: Billing accuracy to GPU·second, 99.99% metering precision
3. **Multi-level isolation**: Shared → Dedicated → Private three-tier, covering solo creators to government clients
4. **Five-tier pricing**: Free tier for acquisition → subscription for retention → enterprise custom for large accounts
5. **Spot + distillation + caching**: Triple cost optimization to bring per-video cost to profitable levels

**First-Year Deployment & Operations Cost** (3 Full Stack + 3 Inference + CDN):

| Category | Annual Cost |
|----------|-------------|
| GPU Clusters (36×H100 + 8×A100) | ~$1.2M (cloud reserved) |
| Storage + Network + CDN | ~$0.4M |
| Personnel (8-10 people) | ~$1.2M |
| Software licenses + Security audits | ~$0.2M |
| **Total** | **~$3.0M/yr** |

**Break-Even Point**: ~3,000 paying users (avg $99/mo) or 50 Enterprise clients ($60K/yr). At 10,000 paying users, gross margin ≈ 65%.
