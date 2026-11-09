<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getKnowledgeGraph, testKnowledgeGraphQuery, getGraphReasoningPath } from '$lib/apis/classifiers';

	const i18n: any = getContext('i18n');

	export let collectionId: string;
	export let knowledgeName: string = '';

	let loading = true;
	let graph: any = { nodes: [], edges: [], available: false };

	let testQuery = '';
	let testing = false;
	let pprResults: any[] = [];

	// Ф6: цепочка логического вывода
	let pathLoading = false;
	let reasoningPath: any = null;

	$: nodeCount = graph.nodes.length;
	$: radius = Math.max(180, nodeCount * 1.2);
	$: viewSize = radius * 2 + 120;

	$: positions = (() => {
		const map: Record<string, { x: number; y: number }> = {};
		graph.nodes.forEach((node: any, index: number) => {
			const angle = (2 * Math.PI * index) / Math.max(nodeCount, 1);
			map[node.id] = {
				x: viewSize / 2 + radius * Math.cos(angle),
				y: viewSize / 2 + radius * Math.sin(angle)
			};
		});
		return map;
	})();

	const loadGraph = async () => {
		loading = true;
		try {
			graph = await getKnowledgeGraph(localStorage.token, collectionId);
			if (!graph?.available && graph?.nodes?.length === 0) {
				toast.info($i18n.t('Graph is empty. Upload files to build the knowledge graph.'));
			}
		} catch (e) {
			toast.error(`${e}`);
			graph = { nodes: [], edges: [], available: false };
		} finally {
			loading = false;
		}
	};

	const runPprTest = async () => {
		if (!testQuery.trim()) return;
		testing = true;
		pprResults = [];
		reasoningPath = null;
		try {
			const result = await testKnowledgeGraphQuery(localStorage.token, collectionId, testQuery.trim(), 10);
			pprResults = result?.results ?? [];
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			testing = false;
		}
	};

	const runReasoningPath = async () => {
		if (!testQuery.trim()) return;
		pathLoading = true;
		reasoningPath = null;
		try {
			reasoningPath = await getGraphReasoningPath(localStorage.token, collectionId, testQuery.trim());
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			pathLoading = false;
		}
	};

	onMount(loadGraph);
</script>

<div class="flex flex-col gap-3">
	{#if loading}
		<div class="py-10 text-center text-sm text-gray-400">{$i18n.t('Loading graph...')}</div>
	{:else if nodeCount === 0}
		<div class="py-10 text-center text-sm text-gray-400">
			{$i18n.t('Knowledge graph is not built yet for this collection.')}
		</div>
	{:else}
		<div class="rounded-xl border border-gray-100 dark:border-gray-850 overflow-auto max-h-[420px] bg-white dark:bg-gray-900">
			<svg width="100%" height="420" viewBox="0 0 {viewSize} {viewSize}" role="img">
				{#each graph.edges as edge (edge.source + edge.target + edge.label)}
					{@const source = positions[edge.source]}
					{@const target = positions[edge.target]}
					{#if source && target}
						<line
							x1={source.x}
							y1={source.y}
							x2={target.x}
							y2={target.y}
							class="stroke-gray-300 dark:stroke-gray-700"
							stroke-width="0.6"
						>
							<title>{edge.label}</title>
						</line>
					{/if}
				{/each}
				{#each graph.nodes as node (node.id)}
					{@const pos = positions[node.id]}
					{#if pos}
						<circle
							cx={pos.x}
							cy={pos.y}
							r={node.kind === 'passage' ? 5 : 3.5}
							class:fill-blue-500={node.kind !== 'passage'}
							class:fill-gray-400={node.kind === 'passage'}
						>
							<title>{node.label}</title>
						</circle>
						{#if node.kind !== 'passage' && nodeCount <= 120}
							<text x={pos.x + 6} y={pos.y + 2} font-size="9" class="fill-gray-500 dark:fill-gray-400">
								{node.label.slice(0, 24)}
							</text>
						{/if}
					{/if}
				{/each}
			</svg>
		</div>
		<div class="text-xs text-gray-400">
			{nodeCount} {$i18n.t('nodes')} · {graph.edges.length} {$i18n.t('edges')} · {$i18n.t('blue: entity, gray: passage')}
		</div>

		<div class="flex gap-2">
			<input
				class="flex-1 text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				placeholder={$i18n.t('Test Personalized PageRank query...')}
				bind:value={testQuery}
				on:keydown={(e) => e.key === 'Enter' && runPprTest()}
			/>
			<button
				class="px-3 py-2 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black disabled:opacity-50"
				disabled={testing}
				on:click={runPprTest}
			>
				{testing ? $i18n.t('Running...') : $i18n.t('Run PPR')}
			</button>
		</div>

		{#if pprResults.length}
			<div class="rounded-xl border border-gray-100 dark:border-gray-850 divide-y divide-gray-100 dark:divide-gray-850 max-h-64 overflow-y-auto">
				{#each pprResults as result, index}
					<div class="px-3 py-2 text-xs">
						<div class="flex justify-between font-medium">
							<span>#{index + 1}</span>
							<span class="text-gray-400">score: {result.score?.toFixed?.(4) ?? result.score}</span>
						</div>
						<div class="text-gray-500 dark:text-gray-400 line-clamp-2">{result.text}</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Ф6: Reasoning Path — доказуемая цепочка вывода по графу -->
		<button
			class="self-start px-3 py-1.5 text-xs rounded-lg bg-gray-100 dark:bg-gray-800 hover:opacity-80 disabled:opacity-50"
			disabled={pathLoading}
			on:click={runReasoningPath}
		>
			{pathLoading ? $i18n.t('Tracing...') : $i18n.t('Show Reasoning Path')}
		</button>

		{#if reasoningPath}
			{#if reasoningPath.available}
				<div class="rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-indigo-50/50 dark:bg-indigo-950/20 p-3">
					<div class="text-xs font-medium mb-2">{$i18n.t('Reasoning path')}</div>
					<div class="flex flex-wrap items-center gap-1 text-xs">
						{#each reasoningPath.steps as step, index}
							<span class="rounded bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 px-1.5 py-0.5">
								{step.from}
							</span>
							{#if step.relation}
								<span class="text-gray-400">—{step.relation}→</span>
							{:else}
								<span class="text-gray-400">→</span>
							{/if}
							{#if index === reasoningPath.steps.length - 1}
								<span class="rounded bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 px-1.5 py-0.5">
									{step.to}
								</span>
							{/if}
						{/each}
					</div>
					{#if reasoningPath.evidence}
						<div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
							<span class="font-medium">{$i18n.t('Evidence')}:</span>
							{reasoningPath.evidence.slice(0, 300)}
						</div>
					{/if}
				</div>
			{:else}
				<div class="text-xs text-gray-400">
					{$i18n.t('Reasoning path unavailable')}: {reasoningPath.reason}
				</div>
			{/if}
		{/if}
	{/if}
</div>
