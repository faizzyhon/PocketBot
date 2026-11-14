<?php
/**
 * Same job as proxy.mjs, for shared PHP hosting (Hostinger, cPanel...). Upload next to your site, put the key in proxy-secrets.php OUTSIDE the web root
 * (or in a folder blocked by .htaccess), and set provider.endpoint in pocketbot.toml to https://your-site/proxy.php .
 *
 * proxy-secrets.php:   <?php return ['api_key' => '...', 'upstream' => 'openai', 'model' => 'gpt-4o-mini', 'origins' => ['https://example.com'], 'per_hour' => 60];
 * Example only: add real authentication and a spending limit at your API vendor before production use.
 */
declare(strict_types=1);
header('Content-Type: application/json; charset=utf-8');
$cfg = require __DIR__ . '/proxy-secrets.php';
function out(int $code, array $body, string $origin = ''): never {
    http_response_code($code);
    if ($origin !== '') { header("Access-Control-Allow-Origin: $origin"); header('Vary: Origin'); header('Access-Control-Allow-Headers: Content-Type'); header('Access-Control-Allow-Methods: POST, OPTIONS'); }
    echo json_encode($body); exit;
}
$origin = $_SERVER['HTTP_ORIGIN'] ?? '';
if (!in_array($origin, $cfg['origins'], true)) out(403, ['error' => 'origin not allowed']);
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') out(204, [], $origin);
if ($_SERVER['REQUEST_METHOD'] !== 'POST') out(405, ['error' => 'POST only'], $origin);

// per-IP hourly limit (a counter file per IP per hour)
$dir = sys_get_temp_dir() . '/pocketbot-rl'; @mkdir($dir, 0700, true);
$f = "$dir/" . sha1(($_SERVER['REMOTE_ADDR'] ?? '0') . date('YmdH')); $n = (int) @file_get_contents($f);
if ($n >= ($cfg['per_hour'] ?? 60)) out(429, ['error' => 'slow down'], $origin);
file_put_contents($f, (string) ($n + 1), LOCK_EX);

$in = json_decode(file_get_contents('php://input') ?: '', true);
$msgs = [];
foreach (array_slice((array) ($in['messages'] ?? []), -8) as $m)
    if (is_array($m) && in_array($m['role'] ?? '', ['system', 'user', 'assistant'], true) && is_string($m['content'] ?? null)) $msgs[] = ['role' => $m['role'], 'content' => mb_substr($m['content'], 0, 4000)];
if (!$msgs) out(400, ['error' => 'no messages'], $origin);
$max = min((int) ($in['max_tokens'] ?? 300), 400);

if (($cfg['upstream'] ?? 'openai') === 'anthropic') {
    $url = 'https://api.anthropic.com/v1/messages';
    $head = ['content-type: application/json', 'x-api-key: ' . $cfg['api_key'], 'anthropic-version: 2023-06-01'];
    $system = implode("\n", array_column(array_filter($msgs, fn($m) => $m['role'] === 'system'), 'content'));
    $body = ['model' => $cfg['model'], 'max_tokens' => $max, 'system' => $system, 'messages' => array_values(array_filter($msgs, fn($m) => $m['role'] !== 'system'))];
} else {
    $url = $cfg['upstream_url'] ?? 'https://api.openai.com/v1/chat/completions';
    $head = ['content-type: application/json', 'authorization: Bearer ' . $cfg['api_key']];
    $body = ['model' => $cfg['model'], 'max_tokens' => $max, 'temperature' => 0.2, 'messages' => $msgs];
}
$ch = curl_init($url);
curl_setopt_array($ch, [CURLOPT_POST => true, CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 30, CURLOPT_HTTPHEADER => $head, CURLOPT_POSTFIELDS => json_encode($body)]);
$raw = curl_exec($ch); $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE); curl_close($ch);
if ($raw === false || $code !== 200) { error_log("[pocketbot-proxy] upstream $code"); out(502, ['error' => 'upstream failed'], $origin); }
$j = json_decode($raw, true); $text = trim((string) ($j['choices'][0]['message']['content'] ?? $j['content'][0]['text'] ?? ''));
out(200, ['reply' => $text, 'choices' => [['message' => ['content' => $text]]]], $origin);
