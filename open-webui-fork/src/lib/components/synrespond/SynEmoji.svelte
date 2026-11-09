<script lang="ts">
	/**
	 * SynEmoji.svelte — Кастомные минималистичные векторные смайлики/эмоции,
	 * отрисованные с нуля специально под состояния модулей synrespond.
	 *
	 * Поддерживаемые состояния (mood):
	 *  - calm: штатная работа, уверенная маршрутизация (p >= tau_high)
	 *  - focused: активный инференс ONNX / поиск Personalized PageRank
	 *  - curious: серая зона дизамбигуации (tau_low <= p < tau_high)
	 *  - tense: повышенное напряжение в голосе (anger_score 0.45..0.74)
	 *  - angry: критический гнев абонента (anger_score >= 0.75, деэскалация)
	 *  - handoff: перевод диалога на живого оператора (HUMAN_HANDOFF)
	 *  - shielded: отражена промпт-инъекция или замаскированы ПДн (PII Vault)
	 *  - hallucination: сработал алерт Ragas (Faithfulness < threshold)
	 *  - optimized: успешный Auto-Tune Optuna / пройден Rollback Gate
	 *  - sleeping: канал или модуль переведен в неактивный режим
	 */
	export let mood:
		| 'calm'
		| 'focused'
		| 'curious'
		| 'tense'
		| 'angry'
		| 'handoff'
		| 'shielded'
		| 'hallucination'
		| 'optimized'
		| 'sleeping' = 'calm';
	export let size: number = 28;
	export let animated: boolean = true;
	export let label: string = '';

	const PALETTE: Record<string, { bg: string; stroke: string; accent: string }> = {
		calm: { bg: 'var(--syn-block-emerald)', stroke: 'var(--syn-emerald)', accent: 'var(--syn-emerald-ink)' },
		focused: { bg: 'var(--syn-block-cobalt)', stroke: 'var(--syn-cobalt)', accent: 'var(--syn-cobalt-ink)' },
		curious: { bg: 'var(--syn-block-amber)', stroke: 'var(--syn-amber)', accent: 'var(--syn-amber-ink)' },
		tense: { bg: 'var(--syn-block-amber)', stroke: 'var(--syn-amber)', accent: 'var(--syn-amber-ink)' },
		angry: { bg: 'var(--syn-block-coral)', stroke: 'var(--syn-coral)', accent: 'var(--syn-coral-ink)' },
		handoff: { bg: 'var(--syn-block-violet)', stroke: 'var(--syn-violet)', accent: 'var(--syn-violet-ink)' },
		shielded: { bg: 'var(--syn-block-cobalt)', stroke: 'var(--syn-cobalt)', accent: 'var(--syn-cobalt-ink)' },
		hallucination: { bg: 'var(--syn-block-coral)', stroke: 'var(--syn-coral)', accent: 'var(--syn-coral-ink)' },
		optimized: { bg: 'var(--syn-block-emerald)', stroke: 'var(--syn-emerald)', accent: 'var(--syn-emerald-ink)' },
		sleeping: { bg: 'var(--syn-bg-elevated)', stroke: 'var(--syn-text-tertiary)', accent: 'var(--syn-text-muted)' }
	};

	$: theme = PALETTE[mood] ?? PALETTE.calm;
</script>

<span
	class="inline-flex shrink-0 items-center justify-center select-none {animated ? 'syn-anim-float' : ''}"
	style="width: {size}px; height: {size}px;"
	role="img"
	aria-label={label || mood}
	title={label || mood}
>
	<svg
		viewBox="0 0 32 32"
		width={size}
		height={size}
		fill="none"
		xmlns="http://www.w3.org/2000/svg"
	>
		<rect
			x="2.5"
			y="2.5"
			width="27"
			height="27"
			rx="9.5"
			fill={theme.bg}
			stroke={theme.stroke}
			stroke-width="1.5"
		/>

		{#if mood === 'calm'}
			<g class={animated ? 'syn-anim-blink' : ''}>
				<circle cx="11.5" cy="13.5" r="1.6" fill={theme.stroke} />
				<circle cx="20.5" cy="13.5" r="1.6" fill={theme.stroke} />
			</g>
			<path
				d="M11.5 19.2C13.1 21.1 18.9 21.1 20.5 19.2"
				stroke={theme.stroke}
				stroke-width="1.75"
				stroke-linecap="round"
			/>
			<circle cx="24.5" cy="7.5" r="1.5" fill={theme.stroke} opacity="0.55" />

		{:else if mood === 'focused'}
			<path d="M9.5 13.5H13.5" stroke={theme.stroke} stroke-width="1.8" stroke-linecap="round" />
			<path d="M18.5 13.5H22.5" stroke={theme.stroke} stroke-width="1.8" stroke-linecap="round" />
			<circle cx="11.5" cy="13.5" r="1.1" fill={theme.accent} />
			<circle cx="20.5" cy="13.5" r="1.1" fill={theme.accent} />
			<path d="M13 20H19" stroke={theme.stroke} stroke-width="1.6" stroke-linecap="round" />
			<circle
				cx="16"
				cy="16"
				r="11.5"
				stroke={theme.stroke}
				stroke-width="1"
				stroke-dasharray="3 4"
				class={animated ? 'syn-anim-pulse' : ''}
				opacity="0.45"
			/>

		{:else if mood === 'curious'}
			<path d="M9.5 10.8L13.2 11.4" stroke={theme.stroke} stroke-width="1.5" stroke-linecap="round" />
			<path d="M18.5 9.8C19.7 9.0 21.3 9.0 22.5 9.8" stroke={theme.stroke} stroke-width="1.5" stroke-linecap="round" />
			<circle cx="11.5" cy="14.2" r="1.5" fill={theme.stroke} />
			<circle cx="20.5" cy="13.6" r="1.7" fill={theme.stroke} />
			<path
				d="M12.5 20.2C14.2 19.3 16.8 20.7 19.2 19.6"
				stroke={theme.stroke}
				stroke-width="1.7"
				stroke-linecap="round"
			/>

		{:else if mood === 'tense'}
			<path d="M9.5 11.5L13.5 12.2" stroke={theme.stroke} stroke-width="1.5" stroke-linecap="round" />
			<path d="M22.5 11.5L18.5 12.2" stroke={theme.stroke} stroke-width="1.5" stroke-linecap="round" />
			<circle cx="11.5" cy="14.5" r="1.4" fill={theme.stroke} />
			<circle cx="20.5" cy="14.5" r="1.4" fill={theme.stroke} />
			<path
				d="M10.5 20.2Q13.2 18.6 16 20.2T21.5 20.2"
				stroke={theme.stroke}
				stroke-width="1.65"
				stroke-linecap="round"
			/>

		{:else if mood === 'angry'}
			<path d="M9.2 10.2L13.8 12.5" stroke={theme.stroke} stroke-width="1.85" stroke-linecap="round" />
			<path d="M22.8 10.2L18.2 12.5" stroke={theme.stroke} stroke-width="1.85" stroke-linecap="round" />
			<circle cx="11.5" cy="14.6" r="1.6" fill={theme.stroke} />
			<circle cx="20.5" cy="14.6" r="1.6" fill={theme.stroke} />
			<path
				d="M11.2 21.2C13.0 18.8 19.0 18.8 20.8 21.2"
				stroke={theme.stroke}
				stroke-width="1.8"
				stroke-linecap="round"
			/>
			<path d="M24.2 6.2L26.2 8.2M26.2 6.2L24.2 8.2" stroke={theme.stroke} stroke-width="1.4" stroke-linecap="round" />

		{:else if mood === 'handoff'}
			<path
				d="M7.5 15.5V13.5C7.5 9.36 11.3 6 16 6C20.7 6 24.5 9.36 24.5 13.5V15.5"
				stroke={theme.stroke}
				stroke-width="1.5"
				stroke-linecap="round"
			/>
			<rect x="6" y="13.5" width="2.8" height="5" rx="1.4" fill={theme.stroke} />
			<rect x="23.2" y="13.5" width="2.8" height="5" rx="1.4" fill={theme.stroke} />
			<circle cx="12" cy="14.5" r="1.4" fill={theme.stroke} />
			<circle cx="20" cy="14.5" r="1.4" fill={theme.stroke} />
			<path d="M12.5 19.3C14 20.8 18 20.8 19.5 19.3" stroke={theme.stroke} stroke-width="1.6" stroke-linecap="round" />

		{:else if mood === 'shielded'}
			<rect
				x="8.5"
				y="11"
				width="15"
				height="4.5"
				rx="2.25"
				fill={theme.bg}
				stroke={theme.stroke}
				stroke-width="1.6"
			/>
			<circle cx="12.2" cy="13.25" r="1.1" fill={theme.stroke} />
			<circle cx="19.8" cy="13.25" r="1.1" fill={theme.stroke} />
			<path d="M13 20.2L15.2 22L19.5 18.5" stroke={theme.stroke} stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" />

		{:else if mood === 'hallucination'}
			<circle cx="11.2" cy="13.5" r="2.3" stroke={theme.stroke} stroke-width="1.5" />
			<circle cx="11.2" cy="13.5" r="0.8" fill={theme.stroke} />
			<rect x="18.5" y="11.5" width="4" height="4" rx="1" stroke={theme.stroke} stroke-width="1.5" />
			<path
				d="M11 20.5C12.8 19 14.5 22 16.2 20.5C17.9 19 19.5 22 21 20.5"
				stroke={theme.stroke}
				stroke-width="1.6"
				stroke-linecap="round"
			/>

		{:else if mood === 'optimized'}
			<path d="M11.5 11.2L13.1 13.5L11.5 15.8L9.9 13.5Z" fill={theme.stroke} />
			<path d="M20.5 11.2L22.1 13.5L20.5 15.8L18.9 13.5Z" fill={theme.stroke} />
			<path
				d="M11 19C12.8 21.6 19.2 21.6 21 19"
				stroke={theme.stroke}
				stroke-width="1.8"
				stroke-linecap="round"
			/>

		{:else}
			<path d="M9.5 14C10.5 15.4 12.5 15.4 13.5 14" stroke={theme.stroke} stroke-width="1.6" stroke-linecap="round" />
			<path d="M18.5 14C19.5 15.4 21.5 15.4 22.5 14" stroke={theme.stroke} stroke-width="1.6" stroke-linecap="round" />
			<path d="M13.5 20H18.5" stroke={theme.stroke} stroke-width="1.5" stroke-linecap="round" />
		{/if}
	</svg>
</span>
