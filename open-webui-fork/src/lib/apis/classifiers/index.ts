import { WEBUI_API_BASE_URL } from '$lib/constants';

// Энкодеры-классификаторы

export const getEncoders = async (token: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const createNewEncoder = async (token: string, encoder: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/create`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(encoder)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateEncoderById = async (token: string, id: string, encoder: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/${id}/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(encoder)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteEncoderById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/${id}/delete`, {
		method: 'DELETE',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const testEncoderInference = async (token: string, form: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/test`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(form)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// Правила маршрутизации интентов

export const getIntentRoutes = async (token: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/routes`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const createNewIntentRoute = async (token: string, route: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/routes/create`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(route)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateIntentRouteById = async (token: string, id: string, route: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/routes/${id}/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(route)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteIntentRouteById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/routes/${id}/delete`, {
		method: 'DELETE',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// Инспектор графа HippoRAG

export const getKnowledgeGraph = async (token: string, collectionId: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/graph/${collectionId}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const testKnowledgeGraphQuery = async (token: string, collectionId: string, query: string, topK = 10) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/graph/${collectionId}/test`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ query, top_k: topK })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// Active Learning (киллер-фича №1)

export const getIntentFeedback = async (token: string, status = 'pending', limit = 200) => {
	let error = null;

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/classifiers/feedback?status=${status}&limit=${limit}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const labelIntentFeedback = async (token: string, id: string, intentName: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/feedback/${id}/label`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ intent_name: intentName })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const dismissIntentFeedback = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/feedback/${id}/dismiss`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const trainFromFeedback = async (token: string, form: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/feedback/train`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(form)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getTrainJobStatus = async (token: string, jobId: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/train/${jobId}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// Reasoning Path (киллер-фича №6)

export const getGraphReasoningPath = async (token: string, collectionId: string, query: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/graph/${collectionId}/path`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ query })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// PERF-UI-02/PERF-CORE-09: горячая перезагрузка и откат ONNX-модели

export const reloadEncoderModel = async (token: string, modelPath: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/${modelPath}/reload`, {
		method: 'POST',
		headers: { authorization: `Bearer ${token}` }
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const rollbackEncoderModel = async (token: string, modelPath: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/encoders/${modelPath}/rollback`, {
		method: 'POST',
		headers: { authorization: `Bearer ${token}` }
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// PERF-UI-04: Agent Copilot — трехконтурные подсказки оператору

export const agentCopilot = async (token: string, form: object) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/classifiers/copilot`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(form)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
