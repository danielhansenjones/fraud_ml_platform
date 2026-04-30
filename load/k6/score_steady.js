import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter } from 'k6/metrics';

const transactionIds = JSON.parse(open('./transaction_ids.json'));
const errorCount = new Counter('errors');

export const options = {
    scenarios: {
        steady: {
            executor: 'constant-arrival-rate',
            rate: 100,
            timeUnit: '1s',
            duration: '10m',
            preAllocatedVUs: 50,
            maxVUs: 200,
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<50', 'p(99)<200'],
        http_req_failed: ['rate<0.005'],
    },
};

export default function () {
    const txId = transactionIds[Math.floor(Math.random() * transactionIds.length)];
    const res = http.post(
        `${__ENV.ROUTER_URL}/score`,
        JSON.stringify({ transaction_id: txId }),
        { headers: { 'Content-Type': 'application/json' } }
    );
    if (!check(res, { 'status 200': (r) => r.status === 200 })) {
        errorCount.add(1);
    }
}
