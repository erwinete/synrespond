<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getRagasConfig, updateRagasConfig } from '$lib/apis/ragas';

	const i18n: any = getContext('i18n');

	let config: any = null;
	let loading = true;
	let health: any = { webrtc: null, sip: null };

	const CHANNELS = [
		{
			key: 'channels.chat.enabled',
			name: 'Text Chat Bot',
			desc: 'Текстовый диалог, SSE-стриминг, Reasoning Path. Работает соло, без speaches и Asterisk.',
			healthUrl: null
		},
		{
			key: 'channels.webrtc.enabled',
			name: 'Browser WebRTC Voice Bot',
			desc: 'Полнодуплексный голос из браузера: WS /api/v1/voice/webrtc/stream, VAD, barge-in. Работает без SIP.',
			healthUrl: null
		},
		{
			key: 'channels.sip.enabled',
			name: 'SIP Telephony Bot (Asterisk)',
			desc: 'Телефония через voice-sip-bridge :8765 (AudioSocket, 8 кГц, филлеры, эскалация).',
			healthUrl: 'http://127.0.0.1:8765/health'
		}
	];

	const loadAll = async () => {
		loading = true;
		try {
			config = await getRagasConfig(localStorage.token).catch(() => null);
			// Канальные ключи приходят через ai-manifest (общий config-дерево)
			const manifest = await fetch('/api/v1/ai-manifest.json', {
				headers: { authorization: `Bearer ${localStorage.token}` }
			}).then((r) => (r.ok ? r.json() : null));
			channelState = manifest?.current_config ?? {};
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	let channelState: Record<string, any> = {};
	let saving = false;

	const toggleChannel = async (key: string) => {
		saving = true;
		try {
			const next = { ...channelState, [key]: !channelState[key] };
			const response = await fetch('/api/v1/mcp', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${localStorage.token}`
				},
				body: JSON.stringify({
					jsonrpc: '2.0',
					id: 1,
					method: 'tools/call',
					params: {
						name: 'platform_configure_modules',
						arguments: { settings: { [key]: next[key] } }
					}
				})
			});
			const data = await response.json();
			if (data.error) throw data.error;
			channelState = next;
			toast.success($i18n.t('Channel updated'));
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			saving = false;
		}
	};

	// Встроенный тестовый звонок WebRTC со страницы настроек
	let testing = false;
	let testLog: string[] = [];
	let ws: WebSocket | null = null;
	let mediaStream: MediaStream | null = null;
	let audioContext: AudioContext | null = null;
	let processor: ScriptProcessorNode | null = null;

	const startTestCall = async () => {
		testing = true;
		testLog = [];
		try {
			mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
			const proto = location.protocol === 'https:' ? 'wss' : 'ws';
			ws = new WebSocket(`${proto}://${location.host}/api/v1/voice/webrtc/stream`);
			ws.binaryType = 'arraybuffer';
			ws.onmessage = (event) => {
				if (typeof event.data === 'string') {
					const msg = JSON.parse(event.data);
					if (msg.type === 'answer') testLog = [...testLog, `bot: ${msg.text}`];
					if (msg.type === 'transcript') testLog = [...testLog, `you: ${msg.text}`];
					if (msg.type === 'ttfb') testLog = [...testLog, `TTFB: ${msg.ms} ms`];
				}
			};
			ws.onopen = () => {
				audioContext = new AudioContext({ sampleRate: 16000 });
				const source = audioContext.createMediaStreamSource(mediaStream!);
				processor = audioContext.createScriptProcessor(4096, 1, 1);
				processor.onaudioprocess = (e) => {
					if (ws?.readyState === WebSocket.OPEN) {
						const pcm = e.inputBuffer.getChannelData(0);
						const buf = new Int16Array(pcm.length);
						for (let i = 0; i < pcm.length; i++) {
							buf[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32768));
						}
						ws.send(buf.buffer);
					}
				};
				source.connect(processor);
				processor.connect(audioContext.destination);
				testLog = [...testLog, 'Говорите — тестовый звонок активен'];
			};
		} catch (e) {
			toast.error(`${e}`);
			testing = false;
		}
	};

	const stopTestCall = () => {
		processor?.disconnect();
		mediaStream?.getTracks().forEach((t) => t.stop());
		audioContext?.close();
		ws?.close();
		processor = null;
		mediaStream = null;
		audioContext = null;
		ws = null;
		testing = false;
		testLog = [...testLog, 'Звонок завершен'];
	};

	onMount(loadAll);
</script>

<div class="mb-3 text-xs text-gray-500">
	{$i18n.t('Three autonomous channel loops. Toggle modules live — no server restart needed.')}
</div>

{#if loading}
	<div class="py-8 text-center text-sm text-gray-400">{$i18n.t('Loading...')}</div>
{:else}
	<div class="grid grid-cols-1 gap-3 lg:grid-cols-3">
		{#each CHANNELS as channel}
			<div class="rounded-xl border border-gray-100 p-4 dark:border-gray-850">
				<div class="flex items-start justify-between">
					<div class="font-medium">{channel.name}</div>
					<button
						type="button"
						class="relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition
						{channelState[channel.key] ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-700'}"
						disabled={saving}
						on:click={() => toggleChannel(channel.key)}
						aria-label={channel.name}
					>
						<span
							class="inline-block size-3.5 transform rounded-full bg-white shadow transition
							{channelState[channel.key] ? 'translate-x-5' : 'translate-x-1'}"
						/>
					</button>
				</div>
				<div class="mt-1 text-xs text-gray-400">{channel.desc}</div>
				<div class="mt-2 flex items-center gap-1.5 text-xs">
					<span
						class="inline-block size-2 rounded-full
						{channelState[channel.key] ? 'bg-green-500' : 'bg-gray-400'}"
					/>
					{channelState[channel.key] ? $i18n.t('Enabled') : $i18n.t('Disabled')}
				</div>
			</div>
		{/each}
	</div>

	<div class="mt-4 rounded-xl border border-gray-100 p-4 dark:border-gray-850">
		<div class="flex items-center justify-between">
			<div class="text-sm font-medium">{$i18n.t('Test WebRTC Call in Browser')}</div>
			<button
				type="button"
				class="rounded-lg px-3 py-1.5 text-xs font-medium
				{testing ? 'bg-red-500 text-white' : 'bg-black text-white dark:bg-white dark:text-black'}"
				on:click={() => (testing ? stopTestCall() : startTestCall())}
			>
				{testing ? $i18n.t('Stop') : $i18n.t('Start test call')}
			</button>
		</div>
		{#if testLog.length}
			<div class="mt-2 max-h-40 overflow-y-auto rounded-lg bg-gray-50 p-2 text-xs dark:bg-gray-900">
				{#each testLog as line}
					<div>{line}</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}
