"""Feed real UploadedFile records through Streamlit's widget deserializer.

AppTest 1.50 lacks a file uploader driver. This adapter uses its actual
MemoryUploadedFileManager and widget protobuf, without mocking st.file_uploader
or the application's image/PDF/data import helpers.
"""
from pathlib import Path
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.proto.WidgetStates_pb2 import WidgetStates


class UploadDriver:
    def __init__(self):
        self.manager=MemoryUploadedFileManager('/mock/upload')
        self.uploads={}

    def bind(self, app):
        original_run=app._run
        def run(widget_state=None,timeout=None):
            state=WidgetStates()
            state.CopyFrom(widget_state if widget_state is not None else app._tree.get_widget_states())
            visible={u.proto.id for u in app.get('file_uploader')}
            for identity,uploaded in self.uploads.items():
                if identity not in visible:
                    continue
                widget=next((w for w in state.widgets if w.id==identity),None)
                if widget is None:
                    widget=state.widgets.add()
                widget.id=identity
                widget.file_uploader_state_value.CopyFrom(uploaded)
            return original_run(state,timeout)
        app._run=run
        return app

    def upload(self, app, label, paths):
        uploader=next(u for u in app.get('file_uploader') if u.proto.label == label)
        widget_id=uploader.proto.id
        state=WidgetStates()
        state.CopyFrom(app._tree.get_widget_states())
        found=next((w for w in state.widgets if w.id==widget_id),None)
        widget=found if found is not None else state.widgets.add()
        widget.id=widget_id
        widget.ClearField('file_uploader_state_value')
        for i,path in enumerate(paths):
            path=Path(path)
            identity=f'{widget_id}-{i}-{path.name}'
            raw=path.read_bytes()
            record=UploadedFileRec(identity,path.name,'application/octet-stream',raw)
            self.manager.add_file('test session id',record)
            info=widget.file_uploader_state_value.uploaded_file_info.add()
            info.file_id=identity
            info.name=path.name
            info.size=len(raw)
        self.uploads[widget_id]=type(widget.file_uploader_state_value)()
        self.uploads[widget_id].CopyFrom(widget.file_uploader_state_value)
        app._run(state,timeout=90)
        return app
