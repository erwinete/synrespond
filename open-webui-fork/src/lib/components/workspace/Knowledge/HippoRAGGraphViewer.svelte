<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getKnowledgeGraph, testKnowledgeGraphQuery, getGraphReasoningPath } from '$lib/apis/classifiers';

	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import '../../synrespond/synrespond-tokens.css';

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

	// PERF-UI-05: интерактивность — масштаб, панорама, фильтр, поиск, инспектор
	let zoom = 1;
	let panX = 0;
	let panY = 0;
	let nodeFilter: 'all' | 'entity' | 'passage' = 'all';
	let nodeSearch = '';
	let selectedNodeId: string | null = null;
	let dragging = false;
	let dragStart = { x: 0, y: 0, panX: 0, panY: 0 };

	$: nodeCount = graph.nodes.length;
	$: entityCount = graph.nodes.filter((n: any) => n.kind !== 'passage').length;

	// Двухкольцевая раскладка (PERF-UI-05): сущности — внутреннее кольцо,
	// passage-чанки — внешняя орбита
	$: layout = (() => {
		const entities = graph.nodes.filter((n: any) => n.kind !== 'passage');
		const passages = graph.nodes.filter((n: any) => n.kind === 'passage');
		const innerRadius = Math.max(120, entities.length * 9);
		const outerRadius = innerRadius + Math.max(110, Math.sqrt(passages.length) * 42);
		const viewSize = outerRadius * 2 + 140;
		const map: Record<string, { x: number; y: number }> = {};
		entities.forEach((node: any, index: number) => {
			const angle = (2 * Math.PI * index) / Math.max(entities.length, 1);
			map[node.id] = {
				x: viewSize / 2 + innerRadius * Math.cos(angle),
				y: viewSize / 2 + innerRadius * Math.sin(angle)
			};
		});
		passages.forEach((node: any, index: number) => {
			const angle = (2 * Math.PI * index) / Math.max(passages.length, 1) + Math.PI / Math.max(passages.length, 2);
			map[node.id] = {
				x: viewSize / 2 + outerRadius * Math.cos(angle),
				y: viewSize / 2 + outerRadius * Math.sin(angle)
			};
		});
		return { map, viewSize };
	})();

	$: positions = layout.map;
	$: viewSize = layout.viewSize;

	$: matchesSearch = (node: any) =>
		!nodeSearch.trim() || String(node.label ?? '').toLowerCase().includes(nodeSearch.trim().toLowerCase());

	$: visibleNodes = graph.nodes.filter(
		(node: any) =>
			(nodeFilter === 'all' || (nodeFilter === 'entity' ? node.kind !== 'passage' : node.kind === 'passage')) &&
			matchesSearch(node)
	);
	$: visibleIds = new Set(visibleNodes.map((n: any) => n.id));

	// Узлы и ребра активного Reasoning Path (подсветка синтетическим контуром)
	$: pathNodeLabels = new Set(
		(reasoningPath?.available ? reasoningPath.steps ?? [] : []).flatMap((s: any) => [s.from, s.to])
	);
	$: pathSteps = reasoningPath?.available ? reasoningPath.steps ?? [] : [];
	$: isPathEdge = (edge: any) => {
		const sourceLabel = String(graph.nodes.find((n: any) => n.id === edge.source)?.label ?? edge.source);
		const targetLabel = String(graph.nodes.find((n: any) => n.id === edge.target)?.label ?? edge.target);
		return pathSteps.some(
			(step: any, i: number) =>
				sourceLabel === step.from &&
				targetLabel === (pathSteps[i + 1]?.to ?? step.to) &&
				(!step.relation || edge.label === step.relation)
		);
	};
	$: isSelected = (node: any) => node.id === selectedNodeId;
	$: selectedNode = graph.nodes.find((n: any) => n.id === selectedNodeId) ?? null;
	$: selectedTriplets = selectedNode
		? graph.edges.filter((e: any) => e.source === selectedNode.id || e.target === selectedNode.id)
		: [];

	const labelOf = (id: string) => {
		const node = graph.nodes.find((n: any) => n.id === id);
		return node?.label ?? id;
	};

	const zoomIn = () => (zoom = Math.min(2.8, zoom + 0.2));
	const zoomOut = () => (zoom = Math.max(0.6, zoom - 0.2));
	const resetView = () => {
		zoom = 1;
		panX = 0;
		panY = 0;
	};

	const onWheelZoom = (event: WheelEvent) => {
		if (!event.ctrlKey && !event.metaKey) return;
		event.preventDefault();
		zoom = Math.min(2.8, Math.max(0.6, zoom - Math.sign(event.deltaY) * 0.15));
	};

	const onPanStart = (event: MouseEvent) => {
		if (event.button !== 0) return;
		dragging = true;
		dragStart = { x: event.clientX, y: event.clientY, panX, panY };
	};
	const onPanMove = (event: MouseEvent) => {
		if (!dragging) return;
		panX = dragStart.panX + (event.clientX - dragStart.x);
		panY = dragStart.panY + (event.clientY - dragStart.y);
	};
	const onPanEnd = () => (dragging = false);

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

<div class="syn-studio flex flex-col gap-3">
	{#if loading}
		<div class="flex flex-col items-center justify-center py-12 gap-3">
			<SynEmoji mood="focused" size={40} />
			<div class="text-xs text-[var(--syn-text-tertiary)]">{$i18n.t('Loading graph...')}</div>
		</div>
	{:else if nodeCount === 0}
		<div class="syn-color-block syn-block-emerald py-10 text-center space-y-2">
			<SynEmoji mood="curious" size={36} animated={false} />
			<div class="text-sm font-medium">{$i18n.t('Knowledge graph is not built yet for this collection.')}</div>
			<p class="text-xs text-[var(--syn-text-tertiary)]">
				{$i18n.t('Upload files to build the knowledge graph.')}
			</p>
		</div>
	{:else}
		<!-- Панель управления графом: фильтр, поиск, масштаб -->
		<div class="flex flex-wrap items-center gap-2">
			<div class="flex items-center gap-1 rounded-xl bg-[var(--syn-bg-elevated)] p-1 text-xs">
				{#each [['all', 'Все'], ['entity', 'Сущности'], ['passage', 'Документы']] as [mode, label]}
					<button
						type="button"
						on:click={() => (nodeFilter = mode)}
						class="rounded-lg px-2.5 py-1 transition {nodeFilter === mode
							? 'bg-[var(--syn-bg-surface)] font-medium shadow-sm'
							: 'text-[var(--syn-text-tertiary)]'}"
					>
						{label}
					</button>
				{/each}
			</div>

			<div class="relative flex-1 min-w-[160px]">
				<span class="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--syn-text-tertiary)]">
					<SynIcon name="search" size={13} />
				</span>
				<input
					class="w-full rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] pl-8 pr-3 py-1.5 text-xs outline-none focus:border-[var(--syn-emerald)] transition"
					placeholder="Поиск сущности по подстроке..."
					bind:value={nodeSearch}
				/>
			</div>

			<div class="flex items-center gap-1">
				<button
					type="button"
					class="rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2 py-1 text-xs font-mono hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={zoomOut}
					aria-label="Уменьшить"
				>
					−
				</button>
				<span class="w-10 text-center font-mono text-[0.68rem] text-[var(--syn-text-tertiary)]">
					{zoom.toFixed(1)}x
				</span>
				<button
					type="button"
					class="rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2 py-1 text-xs font-mono hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={zoomIn}
					aria-label="Увеличить"
				>
					+
				</button>
				<button
					type="button"
					class="rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2 py-1 text-[0.68rem] hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={resetView}
				>
					Сброс
				</button>
			</div>
		</div>

		<!-- SVG-холст с Zoom/Pan -->
		<div
			class="syn-color-block syn-block-emerald overflow-hidden p-0 {dragging ? 'cursor-grabbing' : 'cursor-grab'}"
			on:mousedown={onPanStart}
			on:mousemove={onPanMove}
			on:mouseup={onPanEnd}
			on:mouseleave={onPanEnd}
			on:wheel={onWheelZoom}
			role="application"
			aria-label="Интерактивный граф знаний"
		>
			<svg width="100%" height="420" viewBox="0 0 {viewSize} {viewSize}" role="img">
				<g transform="translate({panX} {panY}) scale({zoom})">
					{#each graph.edges as edge (edge.source + edge.target + edge.label)}
						{@const source = positions[edge.source]}
						{@const target = positions[edge.target]}
						{#if source && target && (visibleIds.has(edge.source) || visibleIds.has(edge.target))}
							<line
								x1={source.x}
								y1={source.y}
								x2={target.x}
								y2={target.y}
								class:stroke-[var(--syn-emerald)]={!isPathEdge(edge)}
								class:stroke-[var(--syn-cobalt)]={isPathEdge(edge)}
								class:syn-anim-dash={isPathEdge(edge)}
								opacity={isPathEdge(edge) ? 0.9 : 0.25}
								stroke-width={isPathEdge(edge) ? 1.4 : 0.6}
							>
								<title>{edge.label}</title>
							</line>
						{/if}
					{/each}
					{#each visibleNodes as node (node.id)}
						{@const pos = positions[node.id]}
						{@const onPath = pathNodeLabels.has(node.label)}
						{#if pos}
							<circle
								cx={pos.x}
								cy={pos.y}
								r={node.kind === 'passage' ? 5 : 3.5}
								class:fill-[var(--syn-emerald)]={node.kind !== 'passage'}
								class:fill-[var(--syn-text-muted)]={node.kind === 'passage'}
								class:stroke-[var(--syn-cobalt)]={isSelected(node)}
								stroke-width={isSelected(node) ? 2.4 : 0}
								role="button"
								tabindex="-1"
								on:click|stopPropagation={() => (selectedNodeId = node.id)}
							>
								<title>{node.label}</title>
							</circle>
							{#if onPath}
								<circle cx={pos.x} cy={pos.y} r={8} class="fill-[var(--syn-emerald)]" opacity="0.18" />
							{/if}
							{#if node.kind !== 'passage' && nodeCount <= 120}
								<text
									x={pos.x + 6}
									y={pos.y + 2}
									font-size="9"
									class="fill-[var(--syn-text-tertiary)] {matchesSearch(node) && nodeSearch ? 'fill-[var(--syn-cobalt)] font-bold' : ''}"
								>
									{node.label.slice(0, 24)}
								</text>
							{/if}
						{/if}
					{/each}
				</g>
			</svg>
		</div>
		<div class="text-xs text-[var(--syn-text-tertiary)]">
			{nodeCount} {$i18n.t('nodes')} ({entityCount} {$i18n.t('entities')}) · {graph.edges.length} {$i18n.t('edges')} ·
			Ctrl+колесо — масштаб, перетаскивание — панорама
		</div>

		<!-- Инспектор выбранного узла: смежные триплеты -->
		{#if selectedNode}
			<div class="syn-color-block syn-block-cobalt space-y-1.5 text-xs">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-1.5 font-semibold">
						<SynIcon name={selectedNode.kind === 'passage' ? 'layers' : 'hipporag'} size={14} />
						{selectedNode.label.slice(0, 80)}
					</div>
					<button
						type="button"
						class="text-[var(--syn-text-tertiary)] hover:text-[var(--syn-text-primary)]"
						on:click={() => (selectedNodeId = null)}
						aria-label="Закрыть инспектор"
					>
						<SynIcon name="close" size={13} />
					</button>
				</div>
				{#if selectedNode.kind === 'passage' && selectedNode.label?.length > 80}
					<p class="line-clamp-4 text-[var(--syn-text-secondary)]">{selectedNode.label}</p>
				{/if}
				<div class="flex flex-wrap gap-1">
					{#each selectedTriplets.slice(0, 24) as triplet}
						<span class="rounded-md border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-1.5 py-0.5 font-mono text-[0.66rem]">
							{labelOf(triplet.source).slice(0, 20)} —{triplet.label}→ {labelOf(triplet.target).slice(0, 20)}
						</span>
					{:else}
						<span class="text-[var(--syn-text-tertiary)]">Связанных ребер нет</span>
					{/each}
				</div>
			</div>
		{/if}

		<div class="flex gap-2">
			<input
				class="flex-1 text-sm rounded-xl px-3.5 py-2 bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] outline-none focus:border-[var(--syn-emerald)] transition"
				placeholder={$i18n.t('Test Personalized PageRank query...')}
				bind:value={testQuery}
				on:keydown={(e) => e.key === 'Enter' && runPprTest()}
			/>
			<button
				class="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-xl bg-[var(--syn-text-primary)] text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
				disabled={testing}
				on:click={runPprTest}
			>
				<SynIcon name="search" size={14} />
				{testing ? $i18n.t('Running...') : $i18n.t('Run PPR')}
			</button>
		</div>

		{#if pprResults.length}
			<div class="syn-color-block divide-y divide-[var(--syn-border-subtle)] max-h-64 overflow-y-auto !py-1">
				{#each pprResults as result, index}
					<div class="px-2 py-2 text-xs">
						<div class="flex justify-between font-medium">
							<span>#{index + 1}</span>
							<span class="font-mono text-[var(--syn-emerald)]">score: {result.score?.toFixed?.(4) ?? result.score}</span>
						</div>
						<div class="text-[var(--syn-text-secondary)] line-clamp-2">{result.text}</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Ф6: Reasoning Path — доказуемая цепочка вывода по графу -->
		<button
			class="inline-flex items-center gap-1.5 self-start px-3 py-1.5 text-xs rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] hover:bg-[var(--syn-bg-elevated)] disabled:opacity-50 transition"
			disabled={pathLoading}
			on:click={runReasoningPath}
		>
			<SynIcon name="hipporag" size={13} />
			{pathLoading ? $i18n.t('Tracing...') : $i18n.t('Show Reasoning Path')}
		</button>

		{#if reasoningPath}
			{#if reasoningPath.available}
				<div class="syn-color-block syn-block-cobalt">
					<div class="text-xs font-semibold mb-2">{$i18n.t('Reasoning path')}</div>
					<div class="flex flex-wrap items-center gap-1 text-xs">
						{#each reasoningPath.steps as step, index}
							<span class="rounded-md bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] px-1.5 py-0.5">
								{step.from}
							</span>
							{#if step.relation}
								<span class="text-[var(--syn-cobalt)] font-mono">—{step.relation}→</span>
							{:else}
								<span class="text-[var(--syn-cobalt)]">→</span>
							{/if}
							{#if index === reasoningPath.steps.length - 1}
								<span class="rounded-md bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] px-1.5 py-0.5">
									{step.to}
								</span>
							{/if}
						{/each}
					</div>
					{#if reasoningPath.evidence}
						<div class="mt-2 text-xs text-[var(--syn-text-secondary)]">
							<span class="font-medium">{$i18n.t('Evidence')}:</span>
							{reasoningPath.evidence.slice(0, 300)}
						</div>
					{/if}
				</div>
			{:else}
				<div class="text-xs text-[var(--syn-text-tertiary)]">
					{$i18n.t('Reasoning path unavailable')}: {reasoningPath.reason}
				</div>
			{/if}
		{/if}
	{/if}
</div>
