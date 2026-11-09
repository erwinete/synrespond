<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext, tick } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getModels } from '$lib/apis';
	import { getKnowledgeBases } from '$lib/apis/knowledge';
	import { getEncoders, getIntentRoutes } from '$lib/apis/classifiers';

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

	let selectedTab = 'matrix';
	let loading = true;
	let showManifestModal = false;

	$: workspaceModels = models.filter((m) => m?.info);

	const loadAll = async () => {
		loading = true;
		try {
			const token = localStorage.token;
			[encoders, routes, models, knowledgeBases] = await Promise.all([
				getEncoders(token).catch(() => []),
				getIntentRoutes(token).catch(() => []),
				getModels(token).catch(() => []),
				getKnowledgeBases(token).catch(() => [])
			]);
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	onMount(async () => {
		await loadAll();
	});
</script>

<div class="mt-0.5 mb-2 gap-y-3 flex flex-col md:flex-row justify-between">
	<div class="flex items-center text-lg font-medium px-1">
		{$i18n.t('Intents & Routing')}
	</div>
</div>

<div
	class="flex gap-1 rounded-[0.75rem] bg-gray-50 dark:bg-gray-900 p-1 mb-3 text-[0.8rem] font-medium shadow-sm"
	id="intents-tabs-container"
>
	<a
		href="#/"
		on:click|preventDefault={() => (selectedTab = 'matrix')}
		class="min-w-fit px-3 py-1 rounded-lg {selectedTab === 'matrix'
			? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow'
			: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'} transition"
	>
		{$i18n.t('Intent Matrix')}
		{routes.length > 0 ? `(${routes.length})` : ''}
	</a>
	<a
		href="#/"
		on:click|preventDefault={() => (selectedTab = 'encoders')}
		class="min-w-fit px-3 py-1 rounded-lg {selectedTab === 'encoders'
			? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow'
			: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'} transition"
	>
		{$i18n.t('Encoder Models')}
		{encoders.length > 0 ? `(${encoders.length})` : ''}
	</a>
	<a
		href="#/"
		on:click|preventDefault={() => (selectedTab = 'learning')}
		class="min-w-fit px-3 py-1 rounded-lg {selectedTab === 'learning'
			? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow'
			: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'} transition"
	>
		{$i18n.t('Active Learning')}
	</a>
	<a
		href="#/"
		on:click|preventDefault={() => (selectedTab = 'channels')}
		class="min-w-fit px-3 py-1 rounded-lg {selectedTab === 'channels'
			? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow'
			: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'} transition"
	>
		{$i18n.t('Channels & Modules')}
	</a>
	<a
		href="#/"
		on:click|preventDefault={() => (selectedTab = 'mcp')}
		class="min-w-fit px-3 py-1 rounded-lg {selectedTab === 'mcp'
			? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow'
			: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'} transition"
	>
		{$i18n.t('MCP & Harnesses')}
	</a>
</div>

{#if selectedTab === 'mcp' || selectedTab === 'channels'}
	<div class="mb-2 flex justify-end">
		<button
			class="px-3.5 py-2 text-sm font-medium bg-black hover:bg-gray-900 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-black rounded-lg transition"
			on:click={() => (showManifestModal = true)}
		>
			{$i18n.t('AI Manifest & Scenario')}
		</button>
	</div>
{/if}

{#if loading}
	<div class="flex justify-center py-8">
		<svg class="animate-spin size-5 text-gray-400" viewBox="0 0 24 24" fill="none">
			<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
			<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
		</svg>
	</div>
{:else if selectedTab === 'matrix'}
	<IntentRouterTable
		routes={routes}
		models={workspaceModels}
		knowledgeBases={knowledgeBases}
		on:change={loadAll}
	/>
{:else if selectedTab === 'learning'}
	<ActiveLearningQueue {encoders} {routes} on:change={loadAll} />
{:else if selectedTab === 'channels'}
	<ChannelRuntimePanel />
{:else if selectedTab === 'mcp'}
	<MCPControlPanel />
{:else}
	<EncoderRegistry {encoders} on:change={loadAll} />
{/if}

<AIManifestEditorModal bind:show={showManifestModal} botModelId="support-bot" />
