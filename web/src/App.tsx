import { Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { MonthProvider } from "./lib/month";
import { CustomerDetailPage } from "./pages/CustomerDetail";
import { CustomersPage } from "./pages/Customers";
import { MonitoringPage } from "./pages/Monitoring";
import { NotFoundPage } from "./pages/NotFound";
import { OverviewPage } from "./pages/Overview";
import { PerformancePage } from "./pages/Performance";
import { PredictionsPage } from "./pages/Predictions";
import { RetentionPage } from "./pages/Retention";

export function App() {
  return (
    <MonthProvider>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<OverviewPage />} />
          <Route path="customers" element={<CustomersPage />} />
          <Route path="customers/:id" element={<CustomerDetailPage />} />
          <Route path="predictions" element={<PredictionsPage />} />
          <Route path="retention" element={<RetentionPage />} />
          <Route path="performance" element={<PerformancePage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </MonthProvider>
  );
}
