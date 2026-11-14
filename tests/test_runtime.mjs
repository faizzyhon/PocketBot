// node tests/test_runtime.mjs     (Node 18+; tests the parts of web/pocketbot.js that do not need a browser)
import assert from 'node:assert/strict';
import { calc, retrieve, terms } from '../web/pocketbot.js';

// calculator: exact, no eval, refuses anything that is not plain arithmetic
assert.equal(calc('2+2'), '2+2 = 4');
assert.equal(calc('what is (12.5 + 7.5) * 3 / 4'), '(12.5 + 7.5) * 3 / 4 = 15');
assert.equal(calc('2^10'), '2^10 = 1024');
assert.equal(calc('2**3**2'), '2^3^2 = 512');          // right-associative power
assert.equal(calc('-3 * -(4+1)'), '-3 * -(4+1) = 15');
assert.equal(calc('17 × 23'), '17 * 23 = 391');
assert.equal(calc('1/3'), '1/3 = 0.3333333333');
assert.equal(calc('10 / 0'), 'Dividing by zero is undefined.');
for (const bad of ['hello', 'open the door', '2 +', '(1+2', 'process.exit()', '1;alert(1)', '9'.repeat(100) + '+1', '5', '2^2000']) assert.equal(calc(bad), null, bad);

// search: finds the right entry, ignores questions with no overlap, honours aliases and weights
const chunks = [
  { k: 'qa', t: 'FAQ', h: 'What are your opening hours? / When are you open?', q: 'What are your opening hours?', x: 'Monday to Saturday 7:00 to 18:00.', w: 1, u: '' },
  { k: 'qa', t: 'FAQ', h: 'Do you deliver?', q: 'Do you deliver?', x: 'Yes, within 5 km for $4.', w: 1, u: '' },
  { t: 'Menu', h: 'Bread', x: 'Sourdough loaf $6.50. Baguette $3.20.', u: '/menu.html' },
];
assert.deepEqual(terms('What are the prices?', { prices: 'price cost' }), ['price', 'cost']);   // stop words dropped, plural stemmed, alias added
assert.equal(retrieve('When do you open?', chunks, { aliases: { open: 'hours' } })[0].c.q, 'What are your opening hours?');
assert.equal(retrieve('Do you deliver to my house?', chunks)[0].c.q, 'Do you deliver?');
assert.equal(retrieve('How much is sourdough?', chunks)[0].c.u, '/menu.html');
assert.deepEqual(retrieve('quantum chromodynamics', chunks), []);
console.log('runtime tests passed');
