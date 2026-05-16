import http from 'k6/http';
import { check } from 'k6';
import { Counter, Trend } from 'k6/metrics';

const transactionIds = JSON.parse(open('./transaction_ids.json'));
const errorCount = new Counter('errors');
const latencyByStage = new Trend('latency_by_stage', true);

export const options = {
    scenarios: {
        breakdown: {
            executor: 'ramping-arrival-rate',
            startRate: 100,
            timeUnit: '1s',
            preAllocatedVUs: 200,
            maxVUs: 2000,
            stages: [
                { target: 1000, duration: '5m' },
                { target: 3000, duration: '5m' },
                { target: 5000, duration: '5m' },
                { target: 5000, duration: '2m' },
            ],
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<200'],
        http_req_failed: [
            { threshold: 'rate<0.05', abortOnFail: true, delayAbortEval: '10s' },
        ],
    },
    summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
    const txId = transactionIds[Math.floor(Math.random() * transactionIds.length)];
    const res = http.post(
        `${__ENV.ROUTER_URL}/score`,
        JSON.stringify({ transaction_id: txId }),
        { headers: { 'Content-Type': 'application/json' } }
    );
    latencyByStage.add(res.timings.duration);
    if (!check(res, { 'status 200': (r) => r.status === 200 })) {
        errorCount.add(1);
    }
}