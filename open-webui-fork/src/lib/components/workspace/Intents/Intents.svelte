<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import '../../synrespond/synrespond-tokens.css';

	import { getModels } from '$lib/apis';
	import { getKnowledgeBases } from '$lib/apis/knowledge';
	import { getEncoders, getIntentRoutes, getIntentFeedback } from '$lib/apis/classifiers';
	import { getRagasAlerts } from '$lib/apis/ragas';

	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import ModuleSettingsDrawer from '../../synrespond/ModuleSettingsDrawer.svelte';

	import IntentRouterTable from './IntentRouterTable.svelte';
	import EncoderRegistry from './EncoderRegistry.svelte';
	import ActiveLearningQueue from './ActiveLearningQueue.svelte';
	import ChannelRuntimePanel from './ChannelRuntimePanel.svelte';
	import MCPControlPanel from './MCPControlPanel.svelte';
	import AIManifestEditorModal from './AIManifestEditorModal.svelte';

	const i18n: any = getContext('i18n');

	let encoders: any[] = [];
	let routes: any[] = [];
	let models: any[] = [];
	let knowledgeBases: any[] = [];
	let pendingFeedback: any[] = [];
	let ragasAlerts: any[] = [];
	let platformSettings: Record<string, any> = {};

	let selectedTab: 'matrix' | 'encoders' | 'learning' | 'channels' | 'mcp' = 'matrix';
	let loading = true;
	let showManifestModal = false;

	let drawerOpen = false;
	let drawerModule: 'channels' | 'guard' | 'encoder' | 'rag' | 'ragas' = 'encoder';

	$: workspaceModels = models.filter((m) => m?.info);
	$: activeChannelsCount = [
		platformSettings['channels.chat.enabled'] ?? true,
		platformSettings['channels.webrtc.enabled'] ?? true,
		platformSettings['channels.sip.enabled'] ?? true
	].filter(Boolean).length;

	const openModuleSettings = (mod: 'channels' | 'guard' | 'encoder' | 'rag' | 'ragas') => {
		drawerModule = mod;
		drawerOpen = true;
	};

	const loadAll = async () => {
		loading = true;
		try {
			const token = localStorage.token;
			const [enc, rts, mdls, kbs, fb, alerts, manifestRes] = await Promise.all([
				getEncoders(token).catch(() => []),
				getIntentRoutes(token).catch(() => []),
				getModels(token).catch(() => []),
				getKnowledgeBases(token).catch(() => []),
				getIntentFeedback(token, 'pending', 100).catch(() => []),
				getRagasAlerts(token, 20).catch(() => []),
				fetch('/api/v1/ai-manifest.json', {
					headers: { authorization: `Bearer ${token}` }
				})
					.then((r) => (r.ok ? r.json() : null))
					.catch(() => null)
			]);
			encoders = enc ?? [];
			routes = rts ?? [];
			models = mdls ?? [];
			knowledgeBases = kbs ?? [];
			pendingFeedback = fb ?? [];
			ragasAlerts = alerts ?? [];
			platformSettings = manifestRes?.current_config ?? {
				'channels.chat.enabled': true,
				'channels.webrtc.enabled': true,
				'channels.sip.enabled': true,
				'intent_classifier.enable': true,
				'intent_classifier.tau_high': 0.85,
				'intent_classifier.tau_low': 0.50,
				'guardrails.prompt_guard': true,
				'guardrails.pii_masking': true,
				'hipporag.enable': true,
				'rag.hybrid_weight_dense': 0.4,
				'rag.hybrid_weight_bm25': 0.2,
				'rag.hybrid_weight_hipporag': 0.4,
				'ragas.enable': true,
				'ragas.live_sample_rate': 0.15,
				'ragas.hallucination_threshold': 0.70
			};
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	onMount(loadAll);
</script>

<div class="syn-studio min-h-full rounded-2xl p-4 md:p-6 space-y-5">
	<header class="flex flex-wrap items-center justify-between gap-4">
		<div class="flex items-center gap-3.5">
			<SynEmoji
				mood={ragasAlerts.length > 0 ? 'hallucination' : pendingFeedback.length > 10 ? 'curious' : 'calm'}
				size={38}
			/>
			<div>
				<div class="flex items-center gap-2">
					<h1 class="text-lg font-semibold tracking-tight">Synrespond Control Studio</h1>
					<span
						class="inline-flex items-center gap-1 rounded-full bg-[var(--syn-block-emerald)] px-2.5 py-0.5 text-[0.68rem] font-medium text-[var(--syn-emerald)]"
					>
						5–20 мс ONNX · 3-Way RAG
					</span>
				</div>
				<p class="text-xs text-[var(--syn-text-tertiary)]">
					Управление маршрутизацией интентов, энкодерами, каналами связи, защитой ПДн и графом знаний
				</p>
			</div>
		</div>

		<div class="flex items-center gap-2">
			<button
				type="button"
				on:click={() => openModuleSettings('encoder')}
				class="inline-flex items-center gap-2 rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3.5 py-2 text-xs font-medium hover:border-[var(--syn-border-strong)] transition"
			>
				<SynIcon name="sliders" size={15} />
				<span>Настройки модулей</span>
			</button>
			<button
				type="button"
				on:click={() => (showManifestModal = true)}
				class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
			>
				<SynIcon name="spark" size={15} />
				<span>AI-Манифест и Симулятор</span>
			</button>
		</div>
	</header>

	<section aria-label="Архитектурный конвейер модулей" class="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-5">
		<button
			type="button"
			on:click={() => openModuleSettings('channels')}
			class="syn-color-block syn-block-violet text-left flex flex-col justify-between gap-2"
		>
			<div class="flex items-center justify-between w-full">
				<span class="inline-flex items-center gap-1.5 text-xs font-semibold">
					<SynIcon name="channels" size={15} />
					<span>1. Каналы</span>
				</span>
				<SynEmoji mood={activeChannelsCount > 0 ? 'calm' : 'sleeping'} size={22} animated={false} />
			</div>
			<div class="text-[0.72rem] text-[var(--syn-text-secondary)]">
				Активно <b>{activeChannelsCount} из 3</b> (Chat · WebRTC · SIP)
			</div>
		</button>

		<button
			type="button"
			on:click={() => openModuleSettings('guard')}
			class="syn-color-block syn-block-coral text-left flex flex-col justify-between gap-2"
		>
			<div class="flex items-center justify-between w-full">
				<span class="inline-flex items-center gap-1.5 text-xs font-semibold">
					<SynIcon name="shield" size={15} />
					<span>2. Защита и ПДн</span>
				</span>
				<SynEmoji
					mood={platformSettings['guardrails.prompt_guard'] ? 'shielded' : 'sleeping'}
					size={22}
					animated={false}
				/>
			</div>
			<div class="text-[0.72rem] text-[var(--syn-text-secondary)]">
				Guard: <b>{platformSettings['guardrails.injection_threshold'] ?? 0.75}</b> · PII Vault:
				<b>{platformSettings['guardrails.pii_masking'] ? 'ВКЛ' : 'ВЫКЛ'}</b>
			</div>
		</button>

		<button
			type="button"
			on:click={() => openModuleSettings('encoder')}
			class="syn-color-block syn-block-cobalt text-left flex flex-col justify-between gap-2"
		>
			<div class="flex items-center justify-between w-full">
				<span class="inline-flex items-center gap-1.5 text-xs font-semibold">
					<SynIcon name="encoder" size={15} />
					<span>3. ONNX Интенты</span>
				</span>
				<SynEmoji
					mood={platformSettings['intent_classifier.enable'] ? 'focused' : 'sleeping'}
					size={22}
					animated={false}
				/>
			</div>
			<div class="text-[0.72rem] text-[var(--syn-text-secondary)]">
				Пороги: <b>{platformSettings['intent_classifier.tau_low'] ?? 0.50}</b> ..
				<b>{platformSettings['intent_classifier.tau_high'] ?? 0.85}</b> · Правил: <b>{routes.length}</b>
			</div>
		</button>

		<button
			type="button"
			on:click={() => openModuleSettings('rag')}
			class="syn-color-block syn-block-emerald text-left flex flex-col justify-between gap-2"
		>
			<div class="flex items-center justify-between w-full">
				<span class="inline-flex items-center gap-1.5 text-xs font-semibold">
					<SynIcon name="hipporag" size={15} />
					<span>4. 3-Way RAG</span>
				</span>
				<SynEmoji
					mood={platformSettings['hipporag.enable'] ? 'optimized' : 'calm'}
					size={22}
					animated={false}
				/>
			</div>
			<div class="text-[0.72rem] text-[var(--syn-text-secondary)]">
				Веса RRF: <b>{platformSettings['rag.hybrid_weight_dense'] ?? 0.4}</b> /
				<b>{platformSettings['rag.hybrid_weight_bm25'] ?? 0.2}</b> /
				<b>{platformSettings['rag.hybrid_weight_hipporag'] ?? 0.4}</b>
			</div>
		</button>

		<button
			type="button"
			on:click={() => openModuleSettings('ragas')}
			class="syn-color-block syn-block-amber text-left flex flex-col justify-between gap-2"
		>
			<div class="flex items-center justify-between w-full">
				<span class="inline-flex items-center gap-1.5 text-xs font-semibold">
					<SynIcon name="ragas" size={15} />
					<span>5. Ragas & QA</span>
				</span>
				<SynEmoji
					mood={ragasAlerts.length > 0 ? 'hallucination' : 'calm'}
					size={22}
					animated={false}
				/>
			</div>
			<div class="text-[0.72rem] text-[var(--syn-text-secondary)]">
				Сэмпл: <b>{((platformSettings['ragas.live_sample_rate'] ?? 0.15) * 100).toFixed(0)}%</b> · Алертов:
				<b>{ragasAlerts.length}</b>
			</div>
		</button>
	</section>

	<nav
		class="flex flex-wrap items-center gap-1.5 rounded-xl bg-[var(--syn-bg-elevated)] p-1.5 text-xs font-medium"
		aria-label="Разделы студии"
	>
		{#each [
			{ id: 'matrix', label: 'Матрица интентов', icon: 'matrix', badge: routes.length },
			{ id: 'encoders', label: 'ONNX Энкодеры', icon: 'encoder', badge: encoders.length },
			{ id: 'learning', label: 'Active Learning', icon: 'flywheel', badge: pendingFeedback.length },
			{ id: 'channels', label: 'Каналы и Тест звонка', icon: 'phone', badge: activeChannelsCount },
			{ id: 'mcp', label: 'MCP и Харнесы', icon: 'mcp', badge: 8 }
		] as item}
			<button
				type="button"
				on:click={() => (selectedTab = item.id)}
				class="inline-flex items-center gap-2 rounded-lg px-3.5 py-2 transition {selectedTab === item.id
					? 'bg-[var(--syn-bg-surface)] text-[var(--syn-text-primary)] shadow-sm'
					: 'text-[var(--syn-text-secondary)] hover:text-[var(--syn-text-primary)]'}"
			>
				<SynIcon name={item.icon} size={15} />
				<span>{item.label}</span>
				{#if item.badge > 0}
					<span
						class="rounded-full bg-[var(--syn-bg-canvas)] px-1.5 py-0.5 text-[0.65rem] font-mono text-[var(--syn-text-secondary)]"
					>
						{item.badge}
					</span>
				{/if}
			</button>
		{/each}
	</nav>

	{#if loading}
		<div class="flex flex-col items-center justify-center py-16 gap-3">
			<SynEmoji mood="focused" size={44} />
			<div class="text-xs text-[var(--syn-text-tertiary)]">Загрузка конфигурации модулей synrespond...</div>
		</div>
	{:else if selectedTab === 'matrix'}
		<IntentRouterTable
			{routes}
			models={workspaceModels}
			{knowledgeBases}
			on:change={loadAll}
		/>
	{:else if selectedTab === 'encoders'}
		<EncoderRegistry {encoders} on:change={loadAll} />
	{:else if selectedTab === 'learning'}
		<ActiveLearningQueue {encoders} {routes} on:change={loadAll} />
	{:else if selectedTab === 'channels'}
		<ChannelRuntimePanel on:openSettings={() => openModuleSettings('channels')} />
	{:else if selectedTab === 'mcp'}
		<MCPControlPanel />
	{/if}
</div>

<ModuleSettingsDrawer
	bind:open={drawerOpen}
	bind:activeModule={drawerModule}
	bind:settings={platformSettings}
	on:saved={loadAll}
/>

<AIManifestEditorModal bind:show={showManifestModal} botModelId="support-bot" />
