<script lang="ts">
	/**
	 * SynIcon.svelte — Авторский набор векторных иконок synrespond (0 внешних библиотек).
	 * Единая оптическая сетка 24×24, штрих 1.65px, скругленные окончания.
	 */
	export let name:
		| 'matrix'
		| 'encoder'
		| 'flywheel'
		| 'channels'
		| 'mcp'
		| 'hipporag'
		| 'ragas'
		| 'shield'
		| 'sliders'
		| 'play'
		| 'stop'
		| 'plus'
		| 'trash'
		| 'check'
		| 'close'
		| 'search'
		| 'spark'
		| 'wave'
		| 'phone'
		| 'terminal'
		| 'copy'
		| 'refresh'
		| 'arrow-right'
		| 'layers' = 'matrix';
	export let size: number = 18;
	export let strokeWidth: number = 1.65;
	export let className: string = '';
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width={strokeWidth}
	stroke-linecap="round"
	stroke-linejoin="round"
	class="inline-block shrink-0 select-none {className}"
	aria-hidden="true"
>
	{#if name === 'matrix'}
		<circle cx="5" cy="12" r="2.2" />
		<circle cx="19" cy="6" r="2" />
		<circle cx="19" cy="12" r="2" />
		<circle cx="19" cy="18" r="2" />
		<path d="M7.2 11.2L17 6.6M7.2 12H17M7.2 12.8L17 17.4" />
	{:else if name === 'encoder'}
		<rect x="4" y="4" width="16" height="16" rx="4" />
		<path d="M9 9H15V15H9Z" />
		<path d="M4 9H9M15 9H20M4 15H9M15 15H20M9 4V9M15 4V9M9 15V20M15 15V20" />
	{:else if name === 'flywheel'}
		<path d="M20 11A8 8 0 0 0 6.3 6.3L4 8.5" />
		<path d="M4 5V8.5H7.5" />
		<path d="M4 13A8 8 0 0 0 17.7 17.7L20 15.5" />
		<path d="M20 19V15.5H16.5" />
		<circle cx="12" cy="12" r="2" />
	{:else if name === 'channels'}
		<rect x="3" y="4" width="18" height="4.5" rx="2" />
		<rect x="3" y="9.75" width="18" height="4.5" rx="2" />
		<rect x="3" y="15.5" width="18" height="4.5" rx="2" />
		<circle cx="6.5" cy="6.25" r="0.9" fill="currentColor" />
		<circle cx="6.5" cy="12" r="0.9" fill="currentColor" />
		<circle cx="6.5" cy="17.75" r="0.9" fill="currentColor" />
	{:else if name === 'mcp'}
		<path d="M7 8L3 12L7 16" />
		<path d="M17 8L21 12L17 16" />
		<circle cx="10" cy="12" r="1.5" />
		<circle cx="14" cy="12" r="1.5" />
		<path d="M11.5 12H12.5" />
	{:else if name === 'hipporag'}
		<circle cx="6" cy="6" r="2.2" />
		<circle cx="18" cy="7" r="2.2" />
		<circle cx="12" cy="18" r="2.5" />
		<circle cx="12" cy="11" r="1.6" fill="currentColor" />
		<path d="M7.8 7.2L10.6 9.8M16.2 8.2L13.4 9.8M12 12.6V15.5M8 6.5L15.8 7M7.2 8L10.8 16" />
	{:else if name === 'ragas'}
		<circle cx="12" cy="12" r="8.5" />
		<path d="M12 3.5V20.5M3.5 12H20.5" opacity="0.45" />
		<polygon points="12,6 17.5,12 12,16.5 7.5,12" />
	{:else if name === 'shield'}
		<path d="M12 3L19 6V11.5C19 16.2 15.9 19.8 12 21C8.1 19.8 5 16.2 5 11.5V6L12 3Z" />
		<path d="M9.5 12L11.2 13.8L14.8 10" />
	{:else if name === 'sliders'}
		<path d="M4 7H10M14 7H20" />
		<circle cx="12" cy="7" r="2" />
		<path d="M4 12H14M18 12H20" />
		<circle cx="16" cy="12" r="2" />
		<path d="M4 17H6M10 17H20" />
		<circle cx="8" cy="17" r="2" />
	{:else if name === 'play'}
		<polygon points="7,5 19,12 7,19" fill="currentColor" stroke="none" />
	{:else if name === 'stop'}
		<rect x="6.5" y="6.5" width="11" height="11" rx="2.2" fill="currentColor" stroke="none" />
	{:else if name === 'plus'}
		<path d="M12 5V19M5 12H19" />
	{:else if name === 'trash'}
		<path d="M4 7H20M9 7V4.5C9 4.2 9.2 4 9.5 4H14.5C14.8 4 15 4.2 15 4.5V7M6.5 7L7.5 19.5C7.6 20.3 8.2 21 9 21H15C15.8 21 16.4 20.3 16.5 19.5L17.5 7" />
		<path d="M10 11V16.5M14 11V16.5" />
	{:else if name === 'check'}
		<path d="M5 12.5L9.5 17L19 7.5" />
	{:else if name === 'close'}
		<path d="M6 6L18 18M18 6L6 18" />
	{:else if name === 'search'}
		<circle cx="11" cy="11" r="6.5" />
		<path d="M16 16L20.5 20.5" />
	{:else if name === 'spark'}
		<path d="M12 3L14.2 9.8L21 12L14.2 14.2L12 21L9.8 14.2L3 12L9.8 9.8L12 3Z" />
	{:else if name === 'wave'}
		<path d="M4 12V12.01M8 8V16M12 5V19M16 8V16M20 11V13" />
	{:else if name === 'phone'}
		<path d="M6.5 4.5H9.5L11 8.5L8.8 10.2C9.9 12.4 11.6 14.1 13.8 15.2L15.5 13L19.5 14.5V17.5C19.5 18.6 18.6 19.5 17.5 19.5C10.3 19.5 4.5 13.7 4.5 6.5C4.5 5.4 5.4 4.5 6.5 4.5Z" />
	{:else if name === 'terminal'}
		<rect x="3" y="4" width="18" height="16" rx="3" />
		<path d="M7.5 9.5L10.5 12L7.5 14.5M12.5 14.5H16.5" />
	{:else if name === 'copy'}
		<rect x="8" y="8" width="12" height="12" rx="2.5" />
		<path d="M16 8V6C16 4.9 15.1 4 14 4H6C4.9 4 4 4.9 4 6V14C4 15.1 4.9 16 6 16H8" />
	{:else if name === 'refresh'}
		<path d="M20 11A8 8 0 0 0 5.5 7.5M4 4V8H8" />
		<path d="M4 13A8 8 0 0 0 18.5 16.5M20 20V16H16" />
	{:else if name === 'arrow-right'}
		<path d="M5 12H19M13.5 6.5L19 12L13.5 17.5" />
	{:else}
		<polygon points="12,3 21,8 12,13 3,8" />
		<path d="M3 12.5L12 17.5L21 12.5M3 16.5L12 21.5L21 16.5" />
	{/if}
</svg>
