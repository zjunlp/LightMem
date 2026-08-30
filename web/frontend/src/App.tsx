import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "@/components/Layout";
import Ask from "@/pages/Ask";
import Ingest from "@/pages/Ingest";
import Settings from "@/pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Ask />} />
        <Route path="/ingest" element={<Ingest />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/ask" element={<Navigate to="/" replace />} />
        <Route
          path="/configure"
          element={<Navigate to="/settings" replace />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
