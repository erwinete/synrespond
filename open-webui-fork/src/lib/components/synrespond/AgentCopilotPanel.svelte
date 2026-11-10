<script lang="ts">
	import { toast } from 'svelte-sonner';
	import SynIcon from '../synrespond/SynIcon.svelte';
	import SynEmoji from '../synrespond/SynEmoji.svelte';
	import { agentCopilot } from '$lib/apis/classifiers';

	export let knowledgeBases: any[] = [];

	let query = 'Клиент спрашивает про условия возврата товара';
	let selectedCollections: string[] = [];
	let topK = 5;
	let loading = false;
	let result: any = null;

	$: rankOf = (meta: any) => meta?.rank_breakdown ?? null;

	const toggleCollection = (id: string) => {
		selectedCollections = selectedCollections.includes(id)
			? selectedCollections.filter((c) => c !== id)
			: [...selectedCollections, id];
	};

	const ask = async () => {
		if (!query.trim() || selectedCollections.length === 0) {
			toast.error('Введите вопрос и выберите хотя бы одну базу знаний');
			return;
		}
		loading = true;
		try {
			result = await agentCopilot(localStorage.token, {
				query: query.trim(),
				collection_ids: selectedCollections,
				top_k: topK
			});
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	const copyCard = async (text: string) => {
		try {
			await navigator.clipboard.writeText(text);
			toast.success('Ответ скопирован в буфер обмена');
		} catch {
			toast.error('Не удалось скопировать текст');
		}
	};
</script>

<div class="space-y-3">
	<div class="syn-color-block syn-block-violet space-y-3">
		<div class="flex items-center gap-3">
			<SynEmoji mood="focused" size={30} animated={false} />
			<div>
				<div class="text-sm font-semibold">Суфлер оператора (Agent Copilot)</div>
				<div class="text-xs text-[var(--syn-text-tertiary)]">
					Трехконтурный поиск Dense + BM25 + HippoRAG с Reasoning Path — за десятки мс, без LLM
				</div>
			</div>
		</div>

		<div class="flex flex-wrap gap-1.5">
			{#each knowledgeBases as kb (kb.id)}
				<button
					type="button"
					on:click={() => toggleCollection(kb.id)}
					class="rounded-lg border px-2.5 py-1 text-xs transition {selectedCollections.includes(kb.id)
						? 'border-[var(--syn-violet)] bg-[var(--syn-bg-surface)] font-medium'
						: 'border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] text-[var(--syn-text-secondary)]'}"
				>
					{kb.name ?? kb.id}
				</button>
			{:else}
				<span class="text-xs text-[var(--syn-text-tertiary)]">Базы знаний не созданы</span>
			{/each}
		</div>

		<div class="flex gap-2">
			<input
				class="flex-1 rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3.5 py-2 text-xs outline-none focus:border-[var(--syn-violet)] transition"
				placeholder="Реплика клиента или вопрос оператора..."
				bind:value={query}
				on:keydown={(e) => e.key === 'Enter' && ask()}
			/>
			<button
				type="button"
				disabled={loading}
				on:click={ask}
				class="inline-flex items-center gap-1.5 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
			>
				<SynIcon name="search" size={14} />
				<span>{loading ? 'Поиск...' : 'Подсказать'}</span>
			</button>
		</div>
	</div>

	{#if result?.reasoning_path?.available}
		<div class="syn-color-block syn-block-cobalt space-y-1.5 text-xs">
			<div class="flex items-center gap-1.5 font-semibold">
				<SynIcon name="hipporag" size={14} />
				Reasoning Path
			</div>
			<div class="flex flex-wrap items-center gap-1">
				{#each result.reasoning_path.steps ?? [] as step, i}
					<span class="rounded-md bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] px-1.5 py-0.5">
						{step.from}
					</span>
					<span class="text-[var(--syn-cobalt)] font-mono">—{step.relation ?? ''}→</span>
					{#if i === (result.reasoning_path.steps ?? []).length - 1}
						<span class="rounded-md bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] px-1.5 py-0.5">
							{step.to}
						</span>
					{/if}
				{/each}
			</div>
		</div>
	{/if}

	{#if result?.cards?.length}
		<div class="space-y-2">
			{#each result.cards as card, i (i)}
				<div class="syn-color-block flex items-start justify-between gap-3">
					<div class="min-w-0">
						<div class="flex flex-wrap items-center gap-2 text-[0.68rem] text-[var(--syn-text-tertiary)]">
							<span class="rounded-md bg-[var(--syn-bg-elevated)] px-1.5 py-0.5 font-mono">#{i + 1}</span>
							{#if card.score}
								<span class="rounded-md bg-[var(--syn-block-emerald)] px-1.5 py-0.5 font-mono text-[var(--syn-emerald-ink)]">
									fusion {Number(card.score).toFixed(4)}
								</span>
							{/if}
							{#if rankOf(card.metadata)}
								<span class="font-mono">
									D:{rankOf(card.metadata).dense_rank ?? '—'} · B:{rankOf(card.metadata).bm25_rank ?? '—'} · H:{rankOf(card.metadata).hipporag_rank ?? '—'}
								</span>
							{/if}
							{#if card.metadata?.retrieval_source}
								<span class="rounded bg-[var(--syn-bg-canvas)] px-1.5 py-0.2 font-mono">{card.metadata.retrieval_source}</span>
							{/if}
						</div>
						<p class="mt-1.5 text-xs leading-relaxed line-clamp-4">{card.text}</p>
					</div>
					<button
						type="button"
						class="inline-flex shrink-0 items-center gap-1 rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1.5 text-xs hover:bg-[var(--syn-bg-elevated)] transition"
						on:click={() => copyCard(card.text)}
					>
						<SynIcon name="copy" size={13} />
						<span>Копировать</span>
					</button>
				</div>
			{/each}
		</div>
	{:else if result}
		<div class="syn-color-block py-8 text-center">
			<SynEmoji mood="curious" size={34} />
			<div class="mt-2 text-xs text-[var(--syn-text-tertiary)]">Релевантных фрагментов не найдено</div>
		</div>
	{/if}
</div>
