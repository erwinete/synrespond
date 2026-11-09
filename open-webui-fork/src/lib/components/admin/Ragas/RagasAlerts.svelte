<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getRagasAlerts } from '$lib/apis/ragas';

	const i18n: any = getContext('i18n');

	let alerts: any[] = [];
	let loading = true;

	const loadAlerts = async () => {
		loading = true;
		try {
			alerts = await getRagasAlerts(localStorage.token);
		} catch (e) {
			toast.error(`${e}`);
			alerts = [];
		} finally {
			loading = false;
		}
	};

	onMount(loadAlerts);
</script>

{#if loading}
	<div class="py-8 text-center text-sm text-gray-400">{$i18n.t('Loading...')}</div>
{:else}
	<div class="flex flex-col gap-2">
		{#each alerts as alert (alert.id)}
			<div class="rounded-xl border border-red-200 bg-red-50/60 px-4 py-3 text-sm dark:border-red-900/50 dark:bg-red-950/20">
				<div class="flex items-start justify-between gap-4">
					<div class="min-w-0">
						<div class="font-medium">{alert.question}</div>
						<div class="mt-1 line-clamp-2 text-xs text-gray-600 dark:text-gray-400">{alert.answer}</div>
						<div class="mt-1 text-[0.65rem] text-gray-400">
							{alert.bot_model_id ?? ''} · {new Date(alert.created_at * 1000).toLocaleString()}
						</div>
					</div>
					<div class="shrink-0 text-xs">
						{#each Object.entries(alert.scores ?? {}) as [key, value]}
							<span class="mr-2">{key}: {typeof value === 'number' ? value.toFixed(2) : value}</span>
						{/each}
					</div>
				</div>
			</div>
		{:else}
			<div class="py-8 text-center text-sm text-gray-400">
				{$i18n.t('No hallucination alerts. Faithfulness is above the threshold.')}
			</div>
		{/each}
	</div>
{/if}
