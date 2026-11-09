<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import {
		createNewEncoder,
		deleteEncoderById,
		testEncoderInference
	} from '$lib/apis/classifiers';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let encoders: any[] = [];

	let showCreateModal = false;
	let newEncoder = { name: '', task: 'sequence_classification', endpoint_url: '', model_path: '' };

	let testText = '';
	let testing = false;
	let testResult: any = null;

	const createEncoder = async () => {
		if (!newEncoder.name.trim() || !newEncoder.model_path.trim()) {
			toast.error($i18n.t('Name and model path are required'));
			return;
		}
		try {
			await createNewEncoder(localStorage.token, {
				...newEncoder,
				name: newEncoder.name.trim(),
				model_path: newEncoder.model_path.trim()
			});
			toast.success($i18n.t('Encoder added'));
			showCreateModal = false;
			newEncoder = { name: '', task: 'sequence_classification', endpoint_url: '', model_path: '' };
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeEncoder = async (encoder: any) => {
		try {
			await deleteEncoderById(localStorage.token, encoder.id);
			toast.success($i18n.t('Encoder deleted'));
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const runTest = async (encoder: any) => {
		if (!testText.trim()) {
			toast.error($i18n.t('Enter a test phrase'));
			return;
		}
		testing = true;
		testResult = null;
		try {
			testResult = await testEncoderInference(localStorage.token, {
				text: testText.trim(),
				model_path: encoder.model_path,
				endpoint_url: encoder.endpoint_url
			});
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			testing = false;
		}
	};

	$: maxCandidateConfidence = testResult?.top_k_candidates?.[0]?.confidence ?? 1;
</script>

<div class="flex items-center justify-between mb-2">
	<div class="text-xs text-gray-500">
		{$i18n.t('ONNX encoders for intent classification. Models live in the encoder-classifier service catalog.')}
	</div>
	<button
		class="px-3.5 py-2 text-sm font-medium bg-black hover:bg-gray-900 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-black rounded-lg transition"
		on:click={() => (showCreateModal = true)}
	>
		+ {$i18n.t('Register Encoder')}
	</button>
</div>

<div class="flex flex-col gap-2">
	{#each encoders as encoder (encoder.id)}
		<div class="rounded-xl border border-gray-100 dark:border-gray-850 p-4">
			<div class="flex items-start justify-between gap-3">
				<div>
					<div class="font-medium">{encoder.name}</div>
					<div class="text-xs text-gray-400">
						{encoder.task} · {encoder.model_path}
						{#if encoder.endpoint_url}
							· {encoder.endpoint_url}
						{/if}
					</div>
				</div>
				<button class="text-xs text-red-500 underline" on:click={() => removeEncoder(encoder)}
					>{$i18n.t('Delete')}</button
				>
			</div>

			<div class="mt-3 flex gap-2">
				<input
					class="flex-1 text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
					placeholder={$i18n.t('Test phrase, e.g. "Where is my order 12345?"')}
					bind:value={testText}
				/>
				<button
					class="px-3 py-2 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black disabled:opacity-50"
					disabled={testing}
					on:click={() => runTest(encoder)}
				>
					{testing ? $i18n.t('Testing...') : $i18n.t('Test Inference')}
				</button>
			</div>
		</div>
	{:else}
		<div class="py-8 text-center text-gray-400 text-sm">
			{$i18n.t('No encoders registered yet.')}
		</div>
	{/each}
</div>

{#if testResult}
	<div class="mt-4 rounded-xl border border-gray-100 dark:border-gray-850 p-4">
		<div class="flex justify-between text-xs text-gray-500 mb-2">
			<span>{$i18n.t('Top-K candidates')}</span>
			<span>{$i18n.t('Latency')}: {testResult.latency_ms} ms · action: {testResult.action}</span>
		</div>
		{#each testResult.top_k_candidates as candidate}
			<div class="flex items-center gap-2 mb-1.5">
				<div class="w-40 truncate text-xs">{candidate.intent_name}</div>
				<div class="flex-1 h-2.5 bg-gray-100 dark:bg-gray-850 rounded-full overflow-hidden">
					<div
						class="h-full bg-blue-500 dark:bg-blue-400 rounded-full"
						style="width: {(candidate.confidence / maxCandidateConfidence) * 100}%"
					/>
				</div>
				<div class="w-14 text-right text-xs">{(candidate.confidence * 100).toFixed(1)}%</div>
			</div>
		{/each}
	</div>
{/if}

<Modal bind:show={showCreateModal} size="sm">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('Register Encoder')}</div>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Display name')}</div>
			<input
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newEncoder.name}
				placeholder="ruBERT Intents v1"
			/>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Task')}</div>
			<select
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newEncoder.task}
			>
				<option value="sequence_classification">sequence_classification</option>
				<option value="zero_shot_nli">zero_shot_nli</option>
				<option value="token_classification">token_classification</option>
			</select>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Model path (catalog folder)')}</div>
			<input
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newEncoder.model_path}
				placeholder="rubert-intents-v1"
			/>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Service URL (optional override)')}</div>
			<input
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newEncoder.endpoint_url}
				placeholder="http://encoder-classifier:8001"
			/>
		</label>

		<div class="flex justify-end gap-2 pt-1">
			<button
				class="px-3 py-1.5 text-sm rounded-lg bg-gray-100 dark:bg-gray-800 hover:opacity-80"
				on:click={() => (showCreateModal = false)}>{$i18n.t('Cancel')}</button
			>
			<button
				class="px-3 py-1.5 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black"
				on:click={createEncoder}>{$i18n.t('Create')}</button
			>
		</div>
	</div>
</Modal>
