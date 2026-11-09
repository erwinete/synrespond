<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	const i18n: any = getContext('i18n');

	let tools: any[] = [];
	let harnesses: any[] = [];
	let loading = true;

	let mcpUrl = '/api/v1/mcp';
	let clipboardCopied = false;

	// Stage 4+: адаптация клиентского конфига под 4 агентных харнеса
	type ClientTarget = 'claude' | 'cursor' | 'zcode' | 'codex';
	let clientTarget: ClientTarget = 'claude';
	let clipboardTarget = '';

	const CLIENT_TARGETS: { id: ClientTarget; label: string; file: string; note: string }[] = [
		{ id: 'claude', label: 'Claude Code', file: '.mcp.json (проект) или ~/.claude.json', note: 'JSON, ключ mcpServers; CLI: claude mcp add' },
		{ id: 'cursor', label: 'Cursor', file: '~/.cursor/mcp.json или .cursor/mcp.json', note: 'JSON, ключ mcpServers; подключение при перезапуске Cursor' },
		{ id: 'zcode', label: 'Zcode', file: '~/.zcode/cli/config.json или <repo>/.zcode/config.json', note: 'JSON, вложенный ключ mcp.servers; серверы подключаются автоматически' },
		{ id: 'codex', label: 'Codex CLI', file: '~/.codex/config.toml', note: 'TOML, секция [mcp_servers.*]; токен через переменную окружения' }
	];

	$: targetSpec = CLIENT_TARGETS.find((c) => c.id === clientTarget) ?? CLIENT_TARGETS[0];

	const buildClientConfig = (target: ClientTarget, origin: string): string => {
		const url = `${origin}${mcpUrl}`;
		const name = 'enterprise-ai-platform';
		if (target === 'codex') {
			return [
				'# ~/.codex/config.toml — Codex CLI',
				`# токен: экспортируйте OPENWEBUI_MCP_TOKEN=<API-ключ Open WebUI>`,
				`[mcp_servers.${name}]`,
				`url = "${url}"`,
				`bearer_token_env_var = "OPENWEBUI_MCP_TOKEN"`,
				'',
				'# альтернатива: codex mcp add ' + name + ' --url ' + url
			].join('\\n');
		}
		if (target === 'zcode') {
			return JSON.stringify(
				{
					mcp: {
						servers: {
							[name]: {
								url,
								headers: { Authorization: 'Bearer <OPEN_WEBUI_API_TOKEN>' }
							}
						}
					}
				},
				null,
				2
			);
		}
		if (target === 'cursor') {
			return JSON.stringify(
				{ mcpServers: { [name]: { url, headers: { Authorization: 'Bearer <OPEN_WEBUI_API_TOKEN>' } } } },
				null,
				2
			);
		}
		// claude
		return JSON.stringify(
			{ mcpServers: { [name]: { type: 'http', url, headers: { Authorization: 'Bearer <OPEN_WEBUI_API_TOKEN>' } } } },
			null,
			2
		);
	};

	const copyClientConfig = async (target: ClientTarget = 'claude') => {
		const config = buildClientConfig(target, location.origin);
		await navigator.clipboard.writeText(config);
		clipboardCopied = true;
		clipboardTarget = target;
		setTimeout(() => (clipboardCopied = false), 2000);
		toast.success($i18n.t('Client config copied'));
	};

	// Создание внешнего харнеса
	let showCreate = false;
	let newHarness = {
		name: '',
		description_for_agents: '',
		transport: 'streamable_http',
		endpoint_or_cmd: '',
		auth_token: '',
		timeout_ms: 5000
	};

	const rpc = async (method: string, params: object = {}) => {
		const response = await fetch('/api/v1/mcp', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				authorization: `Bearer ${localStorage.token}`
			},
			body: JSON.stringify({ jsonrpc: '2.0', id: Date.now(), method, params })
		});
		const data = await response.json();
		if (data.error) throw data.error;
		return data.result;
	};

	const loadAll = async () => {
		loading = true;
		try {
			const [toolsResult, harnessList] = await Promise.all([
				rpc('tools/list'),
				fetch('/api/v1/classifiers/harnesses', {
					headers: { authorization: `Bearer ${localStorage.token}` }
				}).then((r) => (r.ok ? r.json() : []))
			]);
			tools = toolsResult.tools ?? [];
			harnesses = harnessList;
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	const callTool = async (toolName: string) => {
		try {
			const arguments_ =
				toolName === 'platform_inspect_status'
					? {}
					: toolName === 'bot_export_behavior_manifest'
						? { bot_model_id: 'support-bot' }
						: {};
			const result = await rpc('tools/call', { name: toolName, arguments: arguments_ });
			const text = result?.content?.[0]?.text ?? JSON.stringify(result);
			toast.success(text.slice(0, 120) || 'OK');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const discoverTools = async (harnessId: string) => {
		try {
			const response = await fetch(`/api/v1/classifiers/harnesses/${harnessId}/discover`, {
				method: 'POST',
				headers: { authorization: `Bearer ${localStorage.token}` }
			});
			const data = await response.json();
			if (data.error) throw data.error;
			toast.success($i18n.t('Discovered {{count}} tools', { count: data.tools?.length ?? 0 }));
			await loadAll();
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const createHarness = async () => {
		try {
			const response = await fetch('/api/v1/classifiers/harnesses/create', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${localStorage.token}`
				},
				body: JSON.stringify(newHarness)
			});
			const data = await response.json();
			if (!response.ok) throw data.detail ?? data;
			toast.success($i18n.t('Harness created'));
			showCreate = false;
			await loadAll();
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	onMount(loadAll);
</script>

<!-- Platform MCP Server -->
<div class="mb-4 rounded-xl border border-gray-100 p-4 dark:border-gray-850">
	<div class="flex flex-wrap items-center justify-between gap-2">
		<div class="text-sm font-medium">{$i18n.t('Platform MCP Server')}</div>
		<div class="flex items-center gap-1.5">
			<select
				class="rounded-lg border border-gray-100 bg-gray-50 px-2 py-1.5 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={clientTarget}
			>
				{#each CLIENT_TARGETS as target}
					<option value={target.id}>{target.label}</option>
				{/each}
			</select>
			<button
				type="button"
				class="rounded-lg bg-black px-3 py-1.5 text-xs text-white dark:bg-white dark:text-black"
				on:click={() => copyClientConfig(clientTarget)}
			>
				{clipboardCopied && clipboardTarget === clientTarget
					? $i18n.t('Copied!')
					: $i18n.t('Copy client config')}
			</button>
		</div>
	</div>
	<div class="mt-2 rounded-lg bg-gray-50 p-2 text-xs dark:bg-gray-900">
		<div class="text-gray-500">
			<code>{targetSpec.file}</code> — {targetSpec.note}
		</div>
		<pre class="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[0.65rem] text-gray-600 dark:text-gray-300">{buildClientConfig(
			clientTarget,
			typeof location !== 'undefined' ? location.origin : ''
		)}</pre>
	</div>
	<div class="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-400">
		<code class="rounded bg-gray-50 px-2 py-1 dark:bg-gray-900">{mcpUrl}</code>
		<code class="rounded bg-gray-50 px-2 py-1 dark:bg-gray-900">JSON-RPC 2.0</code>
		<a class="underline" href="/llms.txt" target="_blank">llms.txt</a>
		<a class="underline" href="/api/v1/ai-manifest.json" target="_blank">ai-manifest.json</a>
	</div>

	<div class="mt-3 text-xs font-medium text-gray-500">{$i18n.t('MCP Tools (8)')}</div>
	<div class="mt-1 divide-y divide-gray-100 dark:divide-gray-850">
		{#each tools as tool}
			<div class="flex items-center justify-between gap-2 py-1.5">
				<div class="min-w-0">
					<code class="text-xs">{tool.name}</code>
					<div class="truncate text-xs text-gray-400">{tool.description}</div>
				</div>
				<button
					type="button"
					class="shrink-0 rounded border border-gray-200 px-2 py-1 text-xs hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
					on:click={() => callTool(tool.name)}
				>
					{$i18n.t('Test call')}
				</button>
			</div>
		{:else}
			{#if loading}
				<div class="py-2 text-xs text-gray-400">{$i18n.t('Loading...')}</div>
			{/if}
		{/each}
	</div>
</div>

<!-- External MCP Harnesses -->
<div class="rounded-xl border border-gray-100 p-4 dark:border-gray-850">
	<div class="flex items-center justify-between">
		<div class="text-sm font-medium">{$i18n.t('External MCP Harnesses')}</div>
		<button
			type="button"
			class="rounded-lg bg-black px-3 py-1.5 text-xs text-white dark:bg-white dark:text-black"
			on:click={() => (showCreate = !showCreate)}
		>
			+ {$i18n.t('Add Harness')}
		</button>
	</div>

	{#if showCreate}
		<div class="mt-3 grid grid-cols-2 gap-2">
			<input
				class="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				placeholder={$i18n.t('Name, e.g. erp_order_agent')}
				bind:value={newHarness.name}
			/>
			<input
				class="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				placeholder={$i18n.t('Endpoint URL or stdio command')}
				bind:value={newHarness.endpoint_or_cmd}
			/>
			<select
				class="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={newHarness.transport}
			>
				<option value="streamable_http">streamable_http</option>
				<option value="sse">sse</option>
				<option value="stdio">stdio</option>
			</select>
			<input
				class="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				placeholder={$i18n.t('Auth token (optional)')}
				bind:value={newHarness.auth_token}
			/>
			<textarea
				class="col-span-2 h-16 rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				placeholder={$i18n.t('Description for agents — capabilities of this harness')}
				bind:value={newHarness.description_for_agents}
			/>
			<button
				type="button"
				class="col-span-2 rounded-lg bg-black px-3 py-2 text-xs text-white dark:bg-white dark:text-black"
				on:click={createHarness}
			>
				{$i18n.t('Create')}
			</button>
		</div>
	{/if}

	<div class="mt-3 divide-y divide-gray-100 dark:divide-gray-850">
		{#each harnesses as harness}
			<div class="flex items-center justify-between gap-2 py-2">
				<div class="min-w-0">
					<code class="text-xs">{harness.id}</code>
					<div class="truncate text-xs text-gray-400">
						{harness.transport} · {harness.endpoint_or_cmd}
					</div>
				</div>
				<button
					type="button"
					class="shrink-0 rounded border border-gray-200 px-2 py-1 text-xs hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
					on:click={() => discoverTools(harness.id)}
				>
					{$i18n.t('Discover Tools')}
				</button>
			</div>
		{:else}
			<div class="py-3 text-xs text-gray-400">
				{$i18n.t('No external harnesses. Add one to call its tools from intent routes (action MCP_HARNESS).')}
			</div>
		{/each}
	</div>
</div>
