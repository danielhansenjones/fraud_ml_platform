import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const transactionIds = JSON.parse(open('./transaction_ids.json'));
const errorCount = new Counter('errors');

export const options = {
    scenarios: {
        ramp: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s',
            preAllocatedVUs: 50,
            maxVUs: 500,
            stages: [
                { target: 500, duration: '15m' },
                { target: 500, duration: '5m' },
                { target: 0, duration: '2m' },
            ],
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<200'],
        http_req_failed: ['rate<0.01'],
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
