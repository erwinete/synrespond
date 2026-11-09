<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import SynIcon from '../../synrespond/SynIcon.svelte';

	const i18n: any = getContext('i18n');

	export let show = false;
	export let botModelId: string = 'support-bot';

	let manifestText = '';
	let validation: any = null;
	let simulating = false;
	let simulation: any = null;
	let simulateChannel = 'chat';
	let simulateText = 'Где мой заказ 94821?';
	let simulateAnger = 0.0;
	let loading = false;

	const rpc = async (method: string, params: object) => {
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

	const loadManifest = async () => {
		loading = true;
		try {
			const result = await rpc('tools/call', {
				name: 'bot_export_behavior_manifest',
				arguments: { bot_model_id: botModelId }
			});
			const structured = result?.structuredContent ?? {};
			manifestText = JSON.stringify(structured, null, 2);
		} catch (e) {
			manifestText = JSON.stringify(
				{ bot: { model_id: botModelId, name: botModelId, system_prompt: '', channels: ['chat'] }, intent_routes: [] },
				null,
				2
			);
			toast.info($i18n.t('No existing manifest — started from template'));
		} finally {
			loading = false;
		}
	};

	const validate = async () => {
		try {
			const manifest = JSON.parse(manifestText);
			const response = await fetch('/api/v1/classifiers/manifest/validate', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${localStorage.token}`
				},
				body: JSON.stringify(manifest)
			});
			if (!response.ok) throw await response.json();
			validation = await response.json();
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const apply = async (dryRun: boolean) => {
		try {
			const manifest = JSON.parse(manifestText);
			const result = await rpc('tools/call', {
				name: 'bot_apply_behavior_manifest',
				arguments: { manifest, dry_run: dryRun }
			});
			const structured = result?.structuredContent ?? {};
			validation = structured.validation ?? validation;
			if (structured.applied) {
				toast.success($i18n.t('Manifest applied'));
			} else if (dryRun) {
				toast.info($i18n.t('Dry run complete — see validation below'));
			}
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const simulate = async () => {
		simulating = true;
		simulation = null;
		try {
			const result = await rpc('tools/call', {
				name: 'bot_simulate_conversation_turn',
				arguments: {
					bot_model_id: botModelId,
					text: simulateText,
					channel: simulateChannel,
					anger_score: simulateAnger
				}
			});
			simulation = result?.structuredContent ?? {};
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			simulating = false;
		}
	};

	const codeClass =
		'h-64 w-full rounded-lg border border-gray-100 bg-gray-950 px-3 py-2 font-mono text-xs text-green-100 outline-none dark:border-gray-850';
</script>

<Modal bind:show={show} size="lg">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('AI-Ready Manifest & Scenario Architect')}</div>
		<div class="text-xs text-gray-400">
			{$i18n.t('BotBehaviorManifest — декларативный сценарий бота целиком. Линтер проверяет слоты, regex, пороги, каналы и MCP-связи и выдает подсказки для агента.')}
		</div>

		<div class="flex gap-2">
			<button
				type="button"
				class="rounded-lg bg-gray-100 px-3 py-1.5 text-xs hover:opacity-80 dark:bg-gray-800"
				on:click={loadManifest}
			>
				{$i18n.t('Reload from platform')}
			</button>
			<button
				type="button"
				class="rounded-lg bg-gray-100 px-3 py-1.5 text-xs hover:opacity-80 dark:bg-gray-800"
				on:click={validate}
			>
				{$i18n.t('Validate')}
			</button>
			<button
				type="button"
				class="rounded-lg bg-gray-100 px-3 py-1.5 text-xs hover:opacity-80 dark:bg-gray-800"
				on:click={() => apply(true)}
			>
				{$i18n.t('Dry run')}
			</button>
			<button
				type="button"
				class="rounded-lg bg-black px-3 py-1.5 text-xs text-white dark:bg-white dark:text-black"
				on:click={() => apply(false)}
			>
				{$i18n.t('Apply')}
			</button>
		</div>

		<textarea class={codeClass} bind:value={manifestText} spellcheck="false" />

		{#if validation}
			<div class="max-h-40 overflow-y-auto rounded-lg border border-gray-100 p-2 text-xs dark:border-gray-850">
				{#if validation.valid}
					<div class="inline-flex items-center gap-1.5 text-[var(--syn-emerald)]">
						<SynIcon name="check" size={13} />
						{$i18n.t('Manifest is valid')}
					</div>
				{/if}
				{#each validation.errors ?? [] as item}
					<div class="mt-1 text-red-500">
						[{item.code}] {item.message}
						<div class="text-gray-500">→ {item.agent_remediation_hint}</div>
					</div>
				{/each}
				{#each validation.warnings ?? [] as item}
					<div class="mt-1 text-amber-600">
						[{item.code}] {item.message}
						<div class="text-gray-500">→ {item.agent_remediation_hint}</div>
					</div>
				{/each}
			</div>
		{/if}

		<div class="rounded-xl border border-gray-100 p-3 dark:border-gray-850">
			<div class="text-sm font-medium">{$i18n.t('Simulate Turn')}</div>
			<div class="mt-2 flex flex-wrap items-center gap-2">
				<select
					class="rounded-lg border border-gray-100 bg-gray-50 px-2 py-1.5 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
					bind:value={simulateChannel}
				>
					<option value="chat">chat</option>
					<option value="webrtc">webrtc</option>
					<option value="sip">sip</option>
				</select>
				<input
					class="min-w-40 flex-1 rounded-lg border border-gray-100 bg-gray-50 px-3 py-1.5 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
					bind:value={simulateText}
				/>
				<label class="flex items-center gap-1 text-xs text-gray-500">
					anger
					<input type="range" min="0" max="1" step="0.05" bind:value={simulateAnger} class="w-24" />
					{simulateAnger}
				</label>
				<button
					type="button"
					class="rounded-lg bg-black px-3 py-1.5 text-xs text-white dark:bg-white dark:text-black disabled:opacity-50"
					disabled={simulating}
					on:click={simulate}
				>
					{simulating ? $i18n.t('Running...') : $i18n.t('Simulate')}
				</button>
			</div>
			{#if simulation}
				<pre class="mt-2 max-h-48 overflow-auto rounded-lg bg-gray-950 p-2 text-xs text-green-100">{JSON.stringify(
					simulation,
					null,
					2
				)}</pre>
			{/if}
		</div>
	</div>
</Modal>
