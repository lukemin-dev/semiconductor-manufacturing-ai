-- Generated from src.database ORM; API creates the same schema.

CREATE TABLE model_version (
	version VARCHAR(80) NOT NULL, 
	metadata_json JSON NOT NULL, 
	PRIMARY KEY (version)
)

;

CREATE TABLE process_data (
	id SERIAL NOT NULL, 
	sample_id VARCHAR(100) NOT NULL, 
	lot_id VARCHAR(100) NOT NULL, 
	features JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;
CREATE INDEX ix_process_data_lot_id ON process_data (lot_id);
CREATE INDEX ix_process_data_sample_id ON process_data (sample_id);

CREATE TABLE prediction_result (
	id SERIAL NOT NULL, 
	process_id INTEGER NOT NULL, 
	model_version VARCHAR(80) NOT NULL, 
	risk_score FLOAT NOT NULL, 
	predicted_fail INTEGER NOT NULL, 
	result JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(process_id) REFERENCES process_data (id), 
	FOREIGN KEY(model_version) REFERENCES model_version (version)
)

;

CREATE TABLE alert_history (
	id SERIAL NOT NULL, 
	prediction_id INTEGER NOT NULL, 
	status VARCHAR(30) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(prediction_id) REFERENCES prediction_result (id)
)

;