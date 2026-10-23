import { Link } from "react-router-dom";

import { PageHead } from "../components/Shell";
import { Card, EmptyState } from "../components/ui";

export function NotFoundPage() {
  return (
    <>
      <PageHead title="Page not found" month={false} />
      <Card>
        <EmptyState title="There is nothing at this address" action={<Link to="/" className="btn">Back to overview</Link>}>
          Use the navigation to open a section of the console.
        </EmptyState>
      </Card>
    </>
  );
}
