# Seedance 2.5 全球大规模部署与计费系统设计

> 本文定义 Seedance 2.5 作为全球 SaaS/PaaS 产品的部署架构、多租户体系、计费模型和财务系统设计。面向企业客户和开发者提供视频生成 API 服务。

---

## 1. 产品定位与服务模式

### 1.1 三层产品形态

```
┌─────────────────────────────────────────────────────────────────┐
│                    Seedance Cloud                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 1: Web UI (创作者工具)                                │ │
│  │  浏览器端 → 输入 prompt + 参考素材 → 生成 → 下载             │ │
│  │  用户: 内容创作者, 营销人员, 独立艺术家                      │ │
│  │  定价: 订阅制 ($29-299/mo) + 超额按量                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 2: API (开发者平台)                                   │ │
│  │  REST/gRPC API → 生成 → Webhook callback                    │ │
│  │  用户: 创业公司, AI 应用开发者, 视频编辑工具集成             │ │
│  │  定价: 纯按量计费 ($/秒视频), 阶梯折扣                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tier 3: Enterprise (私有部署)                              │ │
│  │  专用集群 → 定制模型 → 数据隔离 → 合规保障                  │ │
│  │  用户: 大型媒体公司, 游戏工作室, 政府/军事                   │ │
│  │  定价: 年度合同 ($500K-5M/yr), 含 SLA + 支持               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 生成规格与定价锚点

| 规格 | 内部代号 | 目标延迟 | 资源消耗 (GPU·s) | 基准定价 (每视频) |
|------|---------|---------|-------------------|------------------|
| **Fast** — 480p, 2s, 16fps, 无音频 | `seedance-fast` | <30s | 8 | $0.08 |
| **Standard** — 720p, 5s, 24fps, 带音频 | `seedance-std` | <60s | 30 | $0.30 |
| **Pro** — 1080p, 10s, 30fps, 带音频 | `seedance-pro` | <120s | 120 | $1.20 |
| **Max** — 4K, 30s, 30fps, 带音频 | `seedance-max` | <180s | 600 | $6.00 |
| **Ultra** — 4K, 60s, 30fps, 带音频, 50 参考输入 | `seedance-ultra` | <360s | 1500 | $15.00 |

---

## 2. 全球部署架构

### 2.1 区域规划

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
   │北美东部  │  │北美西部  │  │ 欧洲    │  │ 亚太    │  │ 中东    │
   │us-east  │  │us-west  │  │eu-west  │  │ap-east  │  │me-east  │
   │Virginia │  │Oregon   │  │Frankfurt│  │Singapore│  │Dubai    │
   │8×H100   │  │4×H100   │  │8×H100   │  │8×H100   │  │4×H100   │
   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
        │             │           │           │             │
        └─────────────┴───────────┴───────────┴─────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ 南美    │  │ 非洲    │  │ 南亚    │
              │sa-east  │  │af-south │  │in-west  │
              │São Paulo│  │Cape Town│  │Mumbai   │
              │CDN only │  │CDN only │  │4×H100   │
              └─────────┘  └─────────┘  └─────────┘
```

| Region | 代码 | GPU 节点 | 用途 | 合规 |
|--------|------|---------|------|------|
| **北美东部** | us-east-1 | 8×H100 | 主训练 + 推理 + 控制面 | SOC2, HIPAA |
| **北美西部** | us-west-2 | 4×H100 | 推理 (西海岸低延迟) | SOC2 |
| **欧洲** | eu-central-1 | 8×H100 | 推理 + 数据驻留 | GDPR, EU AI Act |
| **亚太** | ap-southeast-1 | 8×H100 | 推理 + 亚洲客户 | PDPA (SG) |
| **中东** | me-central-1 | 4×H100 | 推理 | UAE Data Law |
| **南亚** | ap-south-1 | 4×H100 | 推理 (印度市场) | DPDP Act 2023 |
| **南美** | sa-east-1 | CDN only | 边缘缓存 | LGPD (BR) |
| **非洲** | af-south-1 | CDN only | 边缘缓存 | POPIA (ZA) |

### 2.2 区域部署策略

```
Region 部署分级:

┌─────────────────────────────────────────────────────────────────┐
│ Tier 1 Region (Full Stack): us-east, eu-central, ap-southeast  │
│   ├── GPU 推理集群 (8×H100)                                    │
│   ├── 模型注册中心 + 权重存储                                   │
│   ├── 用户数据存储 (数据驻留)                                   │
│   ├── 本地数据库 (PostgreSQL + Redis)                          │
│   ├── 计费计算节点                                             │
│   └── 完整的监控+告警栈                                        │
│                                                                  │
│ Tier 2 Region (Inference Only): us-west, me-central, ap-south  │
│   ├── GPU 推理集群 (4×H100)                                    │
│   ├── 模型缓存 (从 Tier 1 同步)                                │
│   ├── 用户数据不落地 (处理完即删)                               │
│   └── 轻量监控                                                  │
│                                                                  │
│ Tier 3 Region (Edge/CDN Only): sa-east, af-south               │
│   ├── CDN 边缘节点 (Cloudflare / Fastly)                       │
│   ├── 静态资源缓存 (结果视频, Web UI)                          │
│   └── 无 GPU — 请求转发到最近 Tier 1/2                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 跨区域流量路由

```
用户请求
  │
  ▼
┌──────────────────────────────────────────────┐
│            Global Traffic Router              │
│  ┌────────────────────────────────────────┐  │
│  │ DNS Geo-steering (Route53 / Cloudflare) │  │
│  │ 1. 基于用户 IP → 最近 Region            │  │
│  │ 2. Region 健康检查 → 故障转移            │  │
│  │ 3. 数据驻留检查 → GDPR 强制路由          │  │
│  │ 4. 容量感知 → 过载 Region 溢出到邻近     │  │
│  └────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│us-east   │    │eu-central│    │ap-se     │
│正常      │    │正常      │    │过载      │
│→ 本地处理│    │→ 本地处理│    │→ 溢出到  │
│          │    │(GDPR OK) │    │  us-west │
└──────────┘    └──────────┘    └──────────┘
```

**路由规则优先级**:
1. **数据驻留** (最高): GDPR 用户 → 强制 EU Region
2. **延迟最优**: IP → GeoDNS → 最近可用 Region
3. **容量溢出**: 主 Region 队列深度 > N → 溢出到备用 Region
4. **成本优化**: Spot GPU 可用时优先使用, 按需 GPU 兜底

---

## 3. 多租户体系

### 3.1 租户隔离模型

```
┌──────────────────────────────────────────────────────────────┐
│                    Tenant Isolation                           │
│                                                               │
│  Control Plane (共享)                                         │
│    ├── 用户管理 + 认证 (Auth0 / Keycloak)                     │
│    ├── 计费 + 发票 (Stripe / 自定义)                          │
│    ├── 用量计量 (每个请求独立计量)                            │
│    └── API 网关 (Kong / Envoy)                                │
│                                                               │
│  Data Plane (按 Tier 隔离)                                    │
│    ├── Shared (Tier 1): GPU 池共享 + 请求级隔离               │
│    ├── Dedicated (Tier 2): Namespace 隔离 + 专用 GPU 配额     │
│    └── Private (Tier 3): 完全物理隔离 + 专用集群              │
└──────────────────────────────────────────────────────────────┘
```

| 隔离维度 | Shared (Tier 1) | Dedicated (Tier 2) | Private (Tier 3) |
|---------|----------------|-------------------|-----------------|
| **计算** | 共享 GPU 池, 请求级隔离 | K8s Namespace, GPU quota | 专用 GPU 节点 |
| **存储** | 共享 S3 bucket, 用户前缀 | 专用 bucket, IAM 限制 | 专用存储集群 |
| **网络** | 共享 VPC | 独立 VPC, PrivateLink | 独立 VPC + VPN/Direct Connect |
| **数据加密** | 服务端加密 (SSE-S3) | 独立 KMS key | 客户自管 KMS (BYOK) |
| **审计日志** | 共享日志流 | 独立日志流 + 导出 | SIEM 集成 |
| **合规认证** | SOC2 | SOC2 + ISO27001 | 按需定制 (FedRAMP, HIPAA) |

### 3.2 租户生命周期管理

```
Provision → Active → Upgrade/Downgrade → Suspended → Terminated
   │           │            │                │            │
   ▼           ▼            ▼                ▼            ▼
创建租户   正常服务    修改配额/套餐    欠费/违规暂停  数据删除
分配ID     用量计量    热升级(无感知)   保留数据30天   合规擦除
选择Region  SLA监控    通知用户         恢复→Active   审计确认
配额度     账单生成    同步修改配额     永久删除
```

---

## 4. 计费系统设计

### 4.1 定价模型

#### 4.1.1 按量计费 (Pay-as-you-go)

| 计费维度 | 单位 | 单价 | 备注 |
|---------|------|------|------|
| **生成时长** | GPU·秒 | $0.04/GPU·s | 按实际 GPU 占用计费, 秒级计量 |
| **视频输出** | 秒 | 见 1.2 表格 | 基于规格的打包价 |
| **存储** | GB·月 | $0.02 | 生成的视频 + 素材存储 |
| **带宽** | GB 出站 | $0.05 | 下载和 CDN 分发 |

**GPU·秒定价细节** (所有计算统一折算为 H100 等效):

| 操作 | GPU·秒 系数 | 说明 |
|------|-----------|------|
| T5 编码 | 1× | 文本编码 (标准) |
| Coarse 生成 (32fr, 256px) | 8× | Stage B |
| Temporal 扩展 (32→128fr) | 6× | Stage C |
| Spatial SR (256→1080p) | 15× | Stage D (1080p) |
| Spatial SR (256→4K) | 60× | Stage D (4K) |
| Audio 生成 | 4× | Stage E |
| Physics 检查 (PhaseLock) | 1.5× | 可选, 提升物理一致性 |
| 后处理 (编码+水印) | 0.5× | Stage F |

**计费公式**:
```
cost = Σ (operation_gpu_seconds × $0.04 × region_multiplier) + storage + bandwidth

其中 region_multiplier:
  us-east: 1.00 (基准)
  us-west: 1.05
  eu-central: 1.12
  ap-southeast: 1.15
  ap-south: 0.90
  me-central: 1.08
```

#### 4.1.2 订阅套餐

| 套餐 | 月费 | 包含额度 | 超额单价 | 特性 |
|------|------|---------|---------|------|
| **Starter** | $29 | 50 GPU·min (3,000 GPU·s) | $0.05/GPU·s | Standard 规格, 10 并发, 社区支持 |
| **Creator** | $99 | 200 GPU·min (12,000 GPU·s) | $0.045/GPU·s | Pro 规格, 50 并发, 优先队列, email 支持 |
| **Studio** | $299 | 700 GPU·min (42,000 GPU·s) | $0.04/GPU·s | Max 规格, 200 并发, Dedicated 队列, Slack 支持 |
| **Business** | $999 | 2,500 GPU·min (150,000 GPU·s) | $0.035/GPU·s | 全部规格, 500 并发, 99.9% SLA, 专属支持 |
| **Enterprise** | 定制 | 按需 | 协商价 | 私有部署, 定制模型, 99.99% SLA, 数据驻留保证 |

**订阅推荐逻辑**:
```
月用量 < 3,000 GPU·s   → Starter ($29)
月用量 3K-12K GPU·s    → Creator ($99, 省 20%)
月用量 12K-42K GPU·s   → Studio ($299, 省 33%)
月用量 42K-150K GPU·s  → Business ($999, 省 40%)
月用量 > 150K GPU·s    → Enterprise (定制报价, 省 50%+)
```

#### 4.1.3 预付费积分 (C2PA-style)

用于一次性项目或非经常使用场景:
- **$10** → 250 GPU·s (无过期)
- **$50** → 1,375 GPU·s (10% 奖励)
- **$200** → 6,000 GPU·s (20% 奖励)

#### 4.1.4 免费层

| 限制 | 值 |
|------|---|
| 每月免费 GPU·s | 300 (约 5 个 Standard 视频) |
| 输出规格 | Fast + Standard |
| 并发 | 2 |
| 水印 | 强制 (不可移除) |
| 存储 | 7 天自动删除 |

### 4.2 计量架构

```
┌──────────────────────────────────────────────────────────────┐
│                      计量管线 (Metering Pipeline)             │
│                                                               │
│  API Gateway (Envoy/Kong)                                    │
│    │ 每个请求 → Request ID, Tenant ID, Region, Timestamp      │
│    │ 注入 Header: X-Seedance-Tenant, X-Seedance-Tier         │
│    ▼                                                         │
│  GPU Orchestrator                                            │
│    │ 记录 GPU 分配时间 + 释放时间 + GPU 类型                  │
│    │ 记录实际操作: encode, denoise, decode, sr, audio...      │
│    ▼                                                         │
│  Metrics Agent (每 GPU 节点)                                 │
│    │ 实时采集: GPU 利用率, VRAM, 每操作耗时                   │
│    │ 推送 → Kafka topic: seedance.metering.raw               │
│    ▼                                                         │
│  Stream Processor (Apache Flink)                             │
│    │ 窗口聚合 (1min tumbling window)                          │
│    │ 去重 + 异常检测 (负数, 超大值, 重复)                     │
│    │ 按 Tenant × Region × Operation 聚合                      │
│    │ 输出 → Kafka topic: seedance.metering.aggregated         │
│    ▼                                                         │
│  Metering Database (ClickHouse)                              │
│    │ 时序数据: 每 tenant 每秒用量                             │
│    │ 物化视图: 时/日/月 预聚合                                │
│    │ 保留: 原始 90天, 聚合 3年                                │
│    ▼                                                         │
│  Billing Engine (Cron, 每小时跑)                             │
│    │ 读取 ClickHouse → 计算费用 → 写入账单数据库               │
│    │ 超额检测 → 通知 → 自动限流 (可选)                        │
│    │ 月度结算 → 生成发票 → Stripe 扣款                       │
│    ▼                                                         │
│  Billing Database (PostgreSQL)                               │
│    │ 账单, 发票, 付款记录, 信用余额                           │
│    │ 审计不可变日志 (WAL archive → S3)                       │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 计量数据 Schema

```sql
-- ClickHouse: 原始使用事件表
CREATE TABLE metering.usage_events (
    event_id        UUID,
    tenant_id       String,
    request_id      String,
    region          LowCardinality(String),
    operation       LowCardinality(String),  -- 't5_encode','coarse_gen','temporal_ext',...
    gpu_type        LowCardinality(String),  -- 'H100','A100','L40S'
    gpu_count       UInt8,
    gpu_seconds     Decimal64(3),            -- 实际 GPU 秒数
    h100_equivalent Decimal64(3),             -- 归一化为 H100 等效
    billable        Decimal64(3),             -- 计费用量 (扣除免费层后)
    request_status  LowCardinality(String),  -- 'success','failed','cancelled'
    error_code      String,
    timestamp       DateTime64(3),
    received_at     DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (tenant_id, timestamp)
TTL timestamp + INTERVAL 90 DAY;

-- 物化视图: 小时聚合
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

### 4.4 账单与发票

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
│  │ total        │        │ 52,400   │          │              │ │
│  └──────────────┴────────┴──────────┴──────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 支付与发票系统集成

```
┌──────────────────────────────────────────────────────┐
│                 Payment Architecture                  │
│                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │ Stripe       │   │ Internal    │   │ Enterprise   │ │
│  │ (Main)       │   │ Credits     │   │ Invoicing    │ │
│  │              │   │             │   │              │ │
│  │ 信用卡/借记卡│   │ 预付费积分  │   │ NET-30/60    │ │
│  │ Apple/Google │   │ 批量采购    │   │ PO + Invoice │ │
│  │ Pay          │   │ 优惠码      │   │ Wire/ACH     │ │
│  │ PayPal       │   │ 内部转账    │   │ 定制付款     │ │
│  │ 自动续费     │   │             │   │              │ │
│  └──────┬───────┘   └──────┬──────┘   └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            ▼                            │
│                  ┌──────────────────┐                    │
│                  │ Billing Service   │                    │
│                  │                   │                    │
│                  │ 发票生成 (PDF)    │                    │
│                  │ 税率计算 (Stripe  │                    │
│                  │   Tax / Avalara)  │                    │
│                  │ 账单 PDF 归档     │                    │
│                  │ Webhook 通知      │                    │
│                  │ 付款失败重试      │                    │
│                  └──────────────────┘                    │
└──────────────────────────────────────────────────────┘
```

---

## 5. 配额与限流

### 5.1 多级配额体系

```
┌────────────────────────────────────────────────────────────┐
│                   Quota Enforcement                        │
│                                                            │
│  Level 1 — Tenant Quota (用户级别)                         │
│    ├── 月 GPU·s 上限 (订阅套餐决定)                        │
│    ├── 日 GPU·s 上限 (防止异常消费)                        │
│    ├── 并发请求上限 (订阅套餐决定)                         │
│    └── 存储容量上限                                        │
│                                                            │
│  Level 2 — Region Quota (区域级别)                         │
│    ├── 每 Region GPU 容量上限                              │
│    ├── 每 Region 队列深度上限                              │
│    └── 跨 Region 溢出配额                                  │
│                                                            │
│  Level 3 — Global Quota (全局级别)                         │
│    ├── 总 GPU 容量上限                                     │
│    ├── 应急预留容量 (5% for burst)                         │
│    └── 免费层用户总量上限                                  │
└────────────────────────────────────────────────────────────┘
```

### 5.2 限流策略

```yaml
# Envoy Rate Limit Configuration
rate_limits:
  - name: "tenant_per_second"
    domain: "seedance"
    descriptors:
      - key: "tenant_id"
        rate_limit:
          unit: second
          requests_per_unit: 10       # Starter
          requests_per_unit: 50       # Creator
          requests_per_unit: 200      # Studio
          requests_per_unit: 500      # Business

  - name: "tenant_per_day_gpu"
    domain: "seedance"
    descriptors:
      - key: "tenant_id"
        rate_limit:
          unit: day
          requests_per_unit: 3000     # Starter (GPU·s)
          requests_per_unit: 12000    # Creator
          requests_per_unit: 42000    # Studio
          requests_per_unit: 150000   # Business

  - name: "global_concurrent"
    domain: "seedance"
    descriptors:
      - key: "global"
        rate_limit:
          unit: second
          requests_per_unit: 2000     # 全局并发上限

  - name: "free_tier_global"
    domain: "seedance"
    descriptors:
      - key: "tier"
        value: "free"
        rate_limit:
          unit: minute
          requests_per_unit: 100
```

### 5.3 超额处理

```
用量达到阈值:
  │
  ├── 月度额度 80%:  发送提醒邮件 + Dashboard 通知
  ├── 月度额度 95%:  发送告警 + 推荐升级套餐
  ├── 月度额度 100%: 自动切换超额计费 ($0.05/GPU·s)
  │                    或: 软限制 (队列低优先级)
  ├── 超额 150%:     强制限流 (排队, 延迟增加)
  ├── 超额 200%:     暂停服务, 需手动确认或升级
  │
  └── 企业用户:      按合同约定的超额机制
                     通常: 自动扩容 + 月底结算
```

---

## 6. 安全与合规

### 6.1 数据安全架构

```
┌──────────────────────────────────────────────────────────────┐
│                     Data Security                            │
│                                                               │
│  传输中 (TLS 1.3)                                             │
│    ├── 客户端 ↔ API Gateway: mTLS (API users)                │
│    ├── 服务间: mTLS + SPIFFE                                │
│    ├── 跨 Region: WireGuard/IPsec tunnel                     │
│    └── 存储访问: TLS + IAM                                   │
│                                                               │
│  静态加密                                                     │
│    ├── 用户数据 (S3): AES-256-GCM, 每租户独立 KMS key         │
│    ├── 模型权重: AES-256, Region-scoped KMS key              │
│    ├── 日志/指标: AES-256, 服务级 key                        │
│    └── 数据库: TDE (Transparent Data Encryption)             │
│                                                               │
│  使用中 (Confidential Computing, H100)                        │
│    ├── GPU TEE: NVIDIA Confidential Computing (CC mode)      │
│    ├── 内存加密: 防止冷启动攻击                               │
│    └── 证明: 远程证明验证 GPU 固件完整性                      │
│                                                               │
│  Key Management (HashiCorp Vault / AWS KMS)                   │
│    ├── 自动轮换 (90天)                                       │
│    ├── BYOK 支持 (Enterprise only)                           │
│    ├── HSM 硬件保护 (FIPS 140-2 Level 3)                     │
│    └── 审计日志: 每次密钥访问记录                             │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 区域合规矩阵

| 法规 | 适用区域 | 关键要求 | 实现方式 |
|------|---------|---------|---------|
| **GDPR** | EU/EEA | 数据驻留, 删除权, DPA | EU Region 专用, 自动删除 API |
| **EU AI Act** | EU | 高风险 AI 系统透明度 | 内容水印, 生成来源标签 (C2PA) |
| **SOC2 Type II** | 全球 (企业) | 安全/可用性/保密性 | 年度审计, 访问控制, 监控 |
| **CCPA/CPRA** | California | 消费者隐私权 | 数据删除 API, 不出售数据 |
| **LGPD** | Brazil | 类似 GDPR | sa-east 数据驻留 (CDN only) |
| **PDPA** | Singapore | 数据保护 | ap-southeast 合规 |
| **DPDP Act 2023** | India | 数据本地化 | ap-south 数据驻留 |
| **HIPAA** | US Healthcare | 医疗数据保护 | Enterprise only, BAA 签署 |

### 6.3 内容安全

```
输入安全:
  ├── Prompt 注入检测 (LLM-based classifier)
  ├── 参考图像 NSFW 检测
  ├── 禁止内容关键词匹配
  └── 速率异常检测 (DoS 预防)

输出安全:
  ├── 生成视频 NSFW 检测 (视频帧 + CLIP)
  ├── 生成音频有害内容检测
  ├── 隐形水印 (StegaStamp, C2PA 元数据)
  ├── Deepfake 检测指纹
  ├── 版权相似度检测 (vs 训练集)
  └── 儿童安全内容检测 (PhotoDNA/CSAM 哈希)

违规处理:
  ├── 实时阻断: 违反内容策略的请求直接拒绝
  ├── 人工审核队列: 边界案例提交审核
  ├── 申诉流程: 用户可对误判提交申诉
  └── 用户教育: 提供合规使用指南
```

---

## 7. SLA 与服务可靠性

### 7.1 SLA 定义

| 级别 | 月度可用性 | 月度补偿 | P50 延迟 | P95 延迟 | 支持响应 |
|------|-----------|---------|---------|---------|---------|
| **Starter** | 99.0% | 服务积分 (超额) | 无保证 | 无保证 | 48h |
| **Creator** | 99.5% | 10% 账单减免 | <90s (Standard) | <180s | 24h |
| **Studio** | 99.9% | 20% 账单减免 | <60s (Standard) | <120s | 8h |
| **Business** | 99.95% | 30% 账单减免 | <45s (Standard) | <90s | 4h |
| **Enterprise** | 99.99% | 可协商 | 定制 | 定制 | 1h (P1), 4h (P2) |

**SLA 计算公式**:
```
可用性 = (总分钟数 - 不可用分钟数) / 总分钟数 × 100%

不可用 = API 返回 5xx 错误率 > 5%, 持续 ≥ 5 分钟

排除:
  - 计划内维护 (提前 72h 通知)
  - 用户超额导致的限流
  - 第三方依赖 (Stripe, Cloudflare) 故障
  - 不可抗力事件
```

### 7.2 高可用架构

```
每个 Tier 1 Region (us-east, eu-central, ap-southeast):

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
│  │ 对象存储 (11 9's) │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 灾难恢复

| 场景 | RPO | RTO | 恢复策略 |
|------|-----|-----|---------|
| **单 GPU 节点故障** | 0 | <2 min | K8s 自动重新调度 |
| **单 AZ 故障** | 0 | <10 min | 流量切换到另一 AZ |
| **单 Region 故障** | <5 min | <30 min | DNS failover 到备用 Region |
| **数据库损坏** | <5 min | <1 hour | Point-in-time recovery (PITR) |
| **模型权重损坏** | 0 | <15 min | S3 版本恢复 + 重新加载 |
| **全 Region 数据丢失** | <1 hour | <4 hours | 跨 Region 备份恢复 |

---

## 8. 成本优化

### 8.1 GPU 资源优化

| 策略 | 节省 | 实现 |
|------|------|------|
| **Spot/Preemptible GPU** | 40-60% | 推理负载的 70% 使用 Spot, 配合 checkpoint 和快速恢复 |
| **GPU 分时复用 (MIG/MPS)** | 15-30% | H100 MIG 拆分为独立实例, 小任务合用 |
| **模型量化 (FP8/INT8)** | 30-50% | H100 FP8 Tensor Core, 推理时量化 |
| **Batching** | 20-40% | 合并小请求为 batch, 小幅增加延迟大幅提升吞吐 |
| **模型蒸馏** | 60-80% | 200B Teacher → 7B Student (Fast tier) |
| **Caching** | 10-30% | 相同/相似 prompt 结果缓存 (Redis + FAISS 语义去重) |
| **Temporal prediction** | 15-25% | 重复 prompt 预生成 + 缓存 |

### 8.2 存储优化

| 策略 | 节省 | 实现 |
|------|------|------|
| **生命周期策略** | 40-60% | 热→冷自动迁移, Free 层 7 天删除 |
| **视频转码压缩** | 50-70% | 生成原始 ProRes → 分发 H.265/AV1 |
| **去重存储** | 10-20% | 内容寻址存储, 相同内容不重复存 |
| **CDN 缓存** | 80-90% (带宽) | 热点视频 CDN 分发, 避免每次从源站拉取 |

### 8.3 网络优化

| 策略 | 节省 | 实现 |
|------|------|------|
| **压缩** | 40-60% | gRPC 请求/响应 brotli 压缩 |
| **内网传输** | 100% | 跨 AZ→同 Region 内网, 避免公网费用 |
| **Regional CDN** | 50-80% | 静态资源 (Web UI) 部署到边缘节点 |

---

## 9. 关键指标与监控

### 9.1 业务大盘

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
│  │ ⚠  eu-central GPU queue depth  >  200 (threshold: 150)  │ │
│  │ ✓  Stripe payment success rate  =  99.8%                 │ │
│  │ ⚠  Free tier abuse detected: tenant_xxx (3σ above avg)  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 技术大盘

| 指标 | 目标 | 告警阈值 |
|------|------|---------|
| **GPU 利用率** | >75% | <60% (资源浪费) |
| **Model FLOPs Utilization (MFU)** | >50% (H100) | <35% |
| **请求成功率** | >99.9% | <99.5% |
| **P95 端到端延迟** | <120s (Standard) | >180s |
| **计费精度** | >99.99% (vs 实际用量) | 偏差 >0.1% |
| **计量事件丢失率** | <0.01% | >0.1% |
| **免费层滥用检测准确率** | >95% | <90% |
| **安全分类器误报率** | <2% | >5% |

---

## 10. 团队与运营

### 10.1 运行团队

| 角色 | 人数 | 职责 |
|------|------|------|
| **SRE Lead** | 1 | 整体可靠性, 事件管理 |
| **SRE (per region)** | 1 | 各 Region 运维 (on-call rotation) |
| **Platform Engineer** | 2 | K8s, CI/CD, IaC |
| **Security Engineer** | 1 | 安全监控, 渗透测试, 合规审计 |
| **Data/ML Engineer** | 1 | 模型更新, 性能优化 |
| **Billing Engineer** | 1 | 计费系统维护, 财务对账 |
| **Support Engineer** | 2 | 客户工单 (Follow-the-sun) |
| **Total Ops** | **8-10** | |

### 10.2 On-call 策略

```
Follow-the-Sun (24/7 coverage):

  08:00-20:00 UTC+8  →  ap-se SRE (Singapore)
  08:00-20:00 UTC+1  →  eu-central SRE (Frankfurt)
  08:00-20:00 UTC-5  →  us-east SRE (Virginia)

  Weekend: 轮值制, 每 4 周一次
  假日: 双倍薪资 + 补休

  Escalation:
    L1 (SRE)     → 15 min 响应, 60 min 解决或升级
    L2 (Lead)   → 30 min 响应, 2h 解决或升级
    L3 (VP Eng) → 重大事故 (影响 >50% 用户或 >$10K/小时损失)
```

---

## 11. 附: Pricing Calculator API

对外提供的定价查询接口:

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

## 12. 小结

全球部署 Seedance 2.5 作为商业产品，核心挑战是**可靠性 + 计费精准 + 合规**的三角平衡。关键架构决策:

1. **三 Region Full Stack + CDN 全覆盖**: 8 个 Region 覆盖全球，3 个 Full Stack 确保核心能力，边缘 CDN 保障分发
2. **秒级计量 + ClickHouse 时序存储**: 保障计费精度到 GPU·秒级别，99.99% 计量准确率
3. **多级隔离**: Shared → Dedicated → Private 三层，覆盖从独立创作者到政府客户
4. **5 级定价**: 免费层获客 → 订阅制留存 → 企业定制锁定大客户
5. **Spot + 蒸馏 + 缓存**: 三重成本优化将单视频成本压到可盈利水平

**首年部署运营成本** (3 Full Stack + 3 Inference + CDN):

| 类别 | 年成本 |
|------|--------|
| GPU 集群 (36×H100 + 8×A100) | ~$1.2M (云 reserved) |
| 存储 + 网络 + CDN | ~$0.4M |
| 人力 (8-10 人) | ~$1.2M |
| 软件许可 + 安全审计 | ~$0.2M |
| **总计** | **~$3.0M/年** |

**盈亏平衡点**: 约 3,000 付费用户 (平均 $99/mo) 或 50 个 Enterprise 客户 ($60K/yr)。达到 10,000 付费用户时，毛利率约 65%。
