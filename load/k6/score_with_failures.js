import http from 'k6/http';
import { check } from 'k6';
import { Counter, Rate } from 'k6/metrics';

const transactionIds = JSON.parse(open('./transaction_ids.json'));
const errorCount = new Counter('errors');
const errorRate = new Rate('error_rate');

export const options = {
    scenarios: {
        steady_with_failure: {
            executor: 'constant-arrival-rate',
            rate: 100,
            timeUnit: '1s',
            duration: '10m',
            preAllocatedVUs: 50,
            maxVUs: 200,
        },
    },
    thresholds: {
        // Errors allowed during the failure window; we expect recovery after 30s
        http_req_failed: ['rate<0.1'],
    },
};

// Champion container is paused externally at t=5m for 30s via:
//   docker compose pause champion-model && sleep 30 && docker compose unpause champion-model
// This script just records what happens - graceful degradation is validated by
// observing that error_rate spikes during the pause and recovers after.

export default function () {
    const txId = transactionIds[Math.floor(Math.random() * transactionIds.length)];
    const res = http.post(
        `${__ENV.ROUTER_URL}/score`,
        JSON.stringify({ transaction_id: txId }),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2s' }
    );
    const ok = check(res, {
        'status 200 or 503': (r) => r.status === 200 || r.status === 503,
    });
    if (!ok) {
        errorCount.add(1);
    }
    errorRate.add(res.status !== 200);
}
